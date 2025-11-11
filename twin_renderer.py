from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import MediaPipe to get standard pose connections
try:
    import mediapipe as mp
    POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)
except Exception:
    # Fallback: a minimal, reasonable set of connections (if mediapipe not available at render time)
    logger.warning("mediapipe not available; using a minimal fallback connection set.")
    # (start,end) landmark indices based on MediaPipe indexing
    POSE_CONNECTIONS = [
        (11, 13), (13, 15),        # Left arm: shoulder->elbow->wrist
        (12, 14), (14, 16),        # Right arm
        (23, 25), (25, 27),        # Left leg: hip->knee->ankle
        (24, 26), (26, 28),        # Right leg
        (11, 12),                  # shoulders
        (23, 24),                  # hips
        (11, 23), (12, 24),        # torso diagonals
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_norm_frames(landmarks_path: Path) -> List[Dict[int, Tuple[float, float, float, float]]]:
    """Load normalized landmarks JSON and return a list of frame dicts:
       [{idx: (x,y,z,v), ...}, ...]
    """
    p = Path(landmarks_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Normalized landmarks file not found: {p}")
    data = json.loads(p.read_text())

    if "frames" not in data:
        raise ValueError(f"Malformed landmarks JSON (missing 'frames'): {p}")

    frames = []
    for frame in data["frames"]:
        lm = frame.get("lm", {})
        # ensure tuple[float,float,float,float]
        parsed = {}
        for k, arr in lm.items():
            # k may be string in JSON; cast to int
            idx = int(k)
            if not isinstance(arr, (list, tuple)) or len(arr) < 4:
                continue
            x, y, z, v = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            parsed[idx] = (x, y, z, v)
        frames.append(parsed)

    return frames


def _compute_global_bbox(frames: List[Dict[int, Tuple[float, float, float, float]]]):
    """Compute min/max over x,y across all frames to determine a good scale."""
    xs, ys = [], []
    for fr in frames:
        for _, (x, y, _, v) in fr.items():
            if v >= 0.2:
                xs.append(x); ys.append(y)
    if not xs or not ys:
        # fallback box
        return (-1.0, 1.0, -1.5, 1.0)
    return (min(xs), max(xs), min(ys), max(ys))


def _fit_scale_and_center(bbox, canvas_w: int, canvas_h: int, margin_ratio: float = 0.1):
    """Given bbox=(xmin,xmax,ymin,ymax), choose a scale so the figure fits nicely."""
    xmin, xmax, ymin, ymax = bbox
    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)

    # Leave margins
    eff_w = canvas_w * (1.0 - 2 * margin_ratio)
    eff_h = canvas_h * (1.0 - 2 * margin_ratio)

    sx = eff_w / width
    sy = eff_h / height
    scale = min(sx, sy)

    # Target center of bbox maps to canvas center
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    canvas_cx = canvas_w // 2
    canvas_cy = canvas_h // 2

    return scale, (cx, cy), (canvas_cx, canvas_cy)


def _to_px(x: float, y: float, scale: float, src_center: Tuple[float, float], dst_center: Tuple[int, int]):
    """Map normalized coords to canvas pixels. y is assumed positive-down already."""
    cx, cy = src_center
    dcx, dcy = dst_center
    px = int(round((x - cx) * scale + dcx))
    py = int(round((y - cy) * scale + dcy))
    return px, py


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def render_twin_2d(landmarks_path: Path, out_path: Path, fps: int,
                   canvas_size: Tuple[int, int] = (720, 720),
                   joint_radius: int = 3,
                   line_thickness: int = 2) -> None:
    """
    Render a black-background stick-figure video from normalized landmarks.

    landmarks_path: path to *_norm.json produced by pose_extractor.py
    out_path:       where to write MP4
    fps:            frame rate to write
    canvas_size:    (width, height) in pixels
    """
    landmarks_path = Path(landmarks_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = _load_norm_frames(landmarks_path)
    if not frames:
        logger.warning("No frames to render for %s; writing a 1-frame placeholder.", landmarks_path)
        frames = [{}]

    canvas_w, canvas_h = canvas_size
    bbox = _compute_global_bbox(frames)
    scale, src_center, dst_center = _fit_scale_and_center(bbox, canvas_w, canvas_h, margin_ratio=0.12)

    logger.info("Rendering twin2d: %s -> %s (fps=%d, frames=%d, scale=%.1f)",
                landmarks_path.name, out_path.name, fps, len(frames), scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1), (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

    # Colors (BGR)
    line_color = (255, 255, 255)   # white
    joint_color = (200, 200, 200)  # light gray
    text_color = (180, 180, 180)

    try:
        for fi, fr in enumerate(frames):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # Draw bones
            for a, b in POSE_CONNECTIONS:
                if a in fr and b in fr:
                    xa, ya, _, va = fr[a]
                    xb, yb, _, vb = fr[b]
                    if va < 0.2 or vb < 0.2:
                        continue
                    pa = _to_px(xa, ya, scale, src_center, dst_center)
                    pb = _to_px(xb, yb, scale, src_center, dst_center)
                    cv2.line(canvas, pa, pb, line_color, line_thickness, lineType=cv2.LINE_AA)

            # Draw joints
            for idx, (x, y, z, v) in fr.items():
                if v < 0.2:
                    continue
                px, py = _to_px(x, y, scale, src_center, dst_center)
                cv2.circle(canvas, (px, py), joint_radius, joint_color, thickness=-1, lineType=cv2.LINE_AA)

            # Optional label
            cv2.putText(canvas, f"{landmarks_path.stem.replace('_norm','')}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

            writer.write(canvas)

            if (fi + 1) % 60 == 0:
                logger.debug("Rendered %d/%d frames…", fi + 1, len(frames))
    finally:
        writer.release()

    logger.info("Wrote twin2d video -> %s", out_path)
