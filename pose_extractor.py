
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "Failed to import mediapipe. Install with: pip install mediapipe"
    ) from e

logger = logging.getLogger(__name__)

POSE_LM = mp.solutions.pose.PoseLandmark
CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS  # (unused here, handy later)


# ──────────────────────────────────────────────────────────────────────────────
# Small geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Lm:
    x: float
    y: float
    z: float
    visibility: float

def _to_lm_list(mediapipe_landmarks, use_3d: bool) -> List[Lm]:
    """Convert MediaPipe landmarks to a list[Lm]; if 3D not available, z=0."""
    lms = []
    if mediapipe_landmarks is None:
        return lms
    for lm in mediapipe_landmarks.landmark:
        z = getattr(lm, "z", 0.0) if use_3d else 0.0
        lms.append(Lm(float(lm.x), float(lm.y), float(z), float(lm.visibility)))
    return lms

def _pelvis(lms: List[Lm]) -> Optional[Tuple[float, float, float]]:
    try:
        lh = lms[POSE_LM.LEFT_HIP.value]
        rh = lms[POSE_LM.RIGHT_HIP.value]
    except Exception:
        return None
    if lh.visibility < 0.2 or rh.visibility < 0.2:
        return None
    return ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0, (lh.z + rh.z) / 2.0)

def _torso_length(lms: List[Lm]) -> Optional[float]:
    """Distance in normalized image/world space between shoulder-mid and hip-mid."""
    try:
        ls = lms[POSE_LM.LEFT_SHOULDER.value]
        rs = lms[POSE_LM.RIGHT_SHOULDER.value]
        lh = lms[POSE_LM.LEFT_HIP.value]
        rh = lms[POSE_LM.RIGHT_HIP.value]
    except Exception:
        return None
    if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < 0.2:
        return None
    sx, sy, sz = (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0, (ls.z + rs.z) / 2.0
    hx, hy, hz = (lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0, (lh.z + rh.z) / 2.0
    return float(np.linalg.norm([sx - hx, sy - hy, sz - hz]) + 1e-6)

def _estimate_facing_sign(raw_frames, vis_thresh=0.4):
    """Return -1 if overall facing-right, +1 if facing-left (median over frames)."""
    diffs = []
    for lms in raw_frames:
        try:
            ls = lms[POSE_LM.LEFT_SHOULDER.value]
            rs = lms[POSE_LM.RIGHT_SHOULDER.value]
        except Exception:
            continue
        if min(ls.visibility, rs.visibility) < vis_thresh:
            continue
        diffs.append(rs.x - ls.x)
    if not diffs:
        return +1  # fallback
    return -1 if np.median(diffs) >= 0 else 1

def _normalize_frame_fixed(lms, lefty: bool, sign_x_fixed: int):
    pel = _pelvis(lms)
    tl = _torso_length(lms)
    if pel is None or tl is None:
        return {}
    # final sign: global facing sign times lefty flip (if any)
    sign_x = sign_x_fixed * (-1 if lefty else +1)

    out = {}
    for i, lm in enumerate(lms):
        if lm.visibility < 0.2:
            continue
        nx = (lm.x - pel[0]) / tl * sign_x
        ny = (lm.y - pel[1]) / tl
        nz = (lm.z - pel[2]) / tl if lm.z is not None else 0.0
        out[i] = (float(nx), float(ny), float(nz), float(lm.visibility))
    return out

def _ema_smooth_sequence(frames: List[Dict[int, Tuple[float, float, float, float]]],
                         alpha: float = 0.25) -> List[Dict[int, Tuple[float, float, float, float]]]:
    """
    Simple exponential smoothing on per-joint coordinates over time.
    Keeps visibility as-is; smooths x,y,z for indices present in each frame.
    """
    if not frames:
        return frames

    smoothed: List[Dict[int, Tuple[float, float, float, float]]] = []
    prev: Dict[int, Tuple[float, float, float, float]] = {}

    for f in frames:
        outf: Dict[int, Tuple[float, float, float, float]] = {}
        # union of keys to avoid losing joints when they drop in/out
        keys = set(prev.keys()) | set(f.keys())
        for k in keys:
            if k in prev and k in f:
                px, py, pz, pv = prev[k]
                cx, cy, cz, cv = f[k]
                sx = px * (1 - alpha) + cx * alpha
                sy = py * (1 - alpha) + cy * alpha
                sz = pz * (1 - alpha) + cz * alpha
                outf[k] = (float(sx), float(sy), float(sz), float(cv))
            elif k in prev:
                # if joint missing now, decay toward previous (light carry)
                px, py, pz, pv = prev[k]
                sx = px * (1 - alpha)
                sy = py * (1 - alpha)
                sz = pz * (1 - alpha)
                outf[k] = (float(sx), float(sy), float(sz), float(pv))
            else:
                outf[k] = f[k]
        smoothed.append(outf)
        prev = outf
    return smoothed


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def extract_and_normalize_pose(video_path: Path, fps: int, use_3d: bool, lefty: bool) -> Dict:
    """
    Extracts MediaPipe pose landmarks from a video and writes:
      - raw landmarks JSON
      - normalized (pelvis-centered, torso-scaled, unified facing) landmarks JSON
    Returns a summary dict with counts and file paths.

    NOTE: This function expects the run directory layout:
      video_path: runs/<ts>/inputs/<file>
      outputs dir: runs/<ts>/outputs/
    """
    video_path = Path(video_path).resolve()
    outputs_dir = video_path.parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    raw_path = outputs_dir / f"landmarks_{stem}_raw.json"
    norm_path = outputs_dir / f"landmarks_{stem}_norm.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return {
            "frames": 0,
            "avg_visibility": None,
            "landmarks_raw_path": str(raw_path),
            "landmarks_norm_path": str(norm_path),
            "error": f"Could not open video: {video_path}"
        }

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(src_fps / max(fps, 1))), 1)
    logger.info("Pose extraction: %s (src_fps=%.2f → target=%d; stride=%d; use_3d=%s; lefty=%s)",
                video_path.name, src_fps, fps, stride, use_3d, lefty)

    # Init MediaPipe Pose
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    raw_frames: List[List[Lm]] = []
    norm_frames: List[Dict[int, Tuple[float, float, float, float]]] = []

    frame_idx = 0
    kept = 0
    vis_sum = 0.0
    vis_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx % stride) != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # choose 3D or 2D landmarks
            if use_3d and res.pose_world_landmarks:
                lms = _to_lm_list(res.pose_world_landmarks, use_3d=True)
            else:
                lms = _to_lm_list(res.pose_landmarks, use_3d=False)

            raw_frames.append(lms)

            # visibility stats
            if lms:
                for lm in lms:
                    vis_sum += lm.visibility
                vis_count += len(lms)

            # After you finish reading frames into raw_frames:
            sign_x_fixed = _estimate_facing_sign(raw_frames)

            # When creating norm_frames:
            norm = _normalize_frame_fixed(lms, lefty=lefty, sign_x_fixed=sign_x_fixed) if lms else {}
            norm_frames.append(norm)


            kept += 1
            frame_idx += 1

            if kept % 30 == 0:
                logger.debug("Processed %d frames…", kept)
    finally:
        cap.release()
        pose.close()

    # Smooth normalized sequence lightly
    norm_frames_sm = _ema_smooth_sequence(norm_frames, alpha=0.25)

    avg_vis = (vis_sum / vis_count) if vis_count > 0 else None

    # Serialize to JSON
    def lm_to_dict(lm: Lm) -> Dict:
        return {"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility}

    raw_payload = {
        "video": str(video_path),
        "use_3d": bool(use_3d),
        "frames": [
            [lm_to_dict(lm) for lm in lms] for lms in raw_frames
        ],
    }

    # normalized: list[ { "i": frame_index, "lm": { idx: [x,y,z,v], ... } }, ... ]
    norm_payload = {
        "video": str(video_path),
        "normalized": {
            "pelvis_origin": True,
            "torso_scaled": True,
            "unified_facing": True,
            "lefty_flip": bool(lefty),
        },
        "frames": [
            {"i": i, "lm": {int(k): [float(x), float(y), float(z), float(v)]
                            for k, (x, y, z, v) in fr.items()}}
            for i, fr in enumerate(norm_frames_sm)
        ],
    }

    raw_path.write_text(json.dumps(raw_payload, indent=2))
    norm_path.write_text(json.dumps(norm_payload, indent=2))

    logger.info("Wrote raw landmarks → %s", raw_path)
    logger.info("Wrote normalized landmarks → %s", norm_path)
    logger.info("Frames kept: %d | Avg visibility: %s", kept, f"{avg_vis:.3f}" if avg_vis is not None else "n/a")

    return {
        "frames": kept,
        "avg_visibility": avg_vis,
        "landmarks_raw_path": str(raw_path),
        "landmarks_norm_path": str(norm_path),
    }
