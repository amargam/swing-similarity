import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math
import numpy as np

# MediaPipe landmark indices (copied to avoid requiring mediapipe import here)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28

logger = logging.getLogger(__name__)

# --------- Distance helpers (Step 5) ---------
def _path_distances(F_me_w: np.ndarray, F_pro_w: np.ndarray, path: List[Tuple[int, int]],
                    metric: str = "euclidean",
                    trim_fraction: float = 0.0) -> Tuple[List[float], float]:
    """
    Compute distances for each (i,j) along the DTW path.
    - metric: 'euclidean' (default). (Hook for future metrics if needed.)
    - trim_fraction: optionally trim the highest/lowest tails (e.g., 0.05 trims 5% each side)
      to reduce the effect of occasional bad frames.
    Returns (step_dists, avg_dist).
    """
    if not path:
        return [], float("inf")

    if metric != "euclidean":
        logger.warning("Unknown metric '%s'; defaulting to euclidean.", metric)
        metric = "euclidean"

    dists = []
    Ti, Tj = F_me_w.shape[0], F_pro_w.shape[0]
    for (i, j) in path:
        if 0 <= i < Ti and 0 <= j < Tj:
            dists.append(float(np.linalg.norm(F_me_w[i] - F_pro_w[j])))

    if not dists:
        return [], float("inf")

    # Optional trimming for robustness
    trim_fraction = max(0.0, min(0.45, float(trim_fraction)))  # keep at least 10% data
    if trim_fraction > 0.0 and len(dists) >= 20:
        lo = np.quantile(dists, trim_fraction)
        hi = np.quantile(dists, 1.0 - trim_fraction)
        dists = [d for d in dists if lo <= d <= hi]

    avg = float(np.mean(dists)) if dists else float("inf")
    return dists, avg


# --------- DTW helpers (Step 4) ---------
def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _dtw_windowed(A: np.ndarray, B: np.ndarray, radius: int) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Classic dynamic-programming DTW with a Sakoe–Chiba band of given radius (in frames).
    Returns (total_cost, path). Path is a list of (i, j) indices from start->end.
    """
    n, m = A.shape[0], B.shape[0]
    r = max(0, int(radius))

    # Use +inf to mark disallowed cells outside the band
    INF = 1e18
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0

    # Backpointers for path reconstruction
    phi_i = np.full((n + 1, m + 1), -1, dtype=np.int32)
    phi_j = np.full((n + 1, m + 1), -1, dtype=np.int32)

    for i in range(1, n + 1):
        j_start = max(1, i - r)
        j_end   = min(m, i + r)
        for j in range(j_start, j_end + 1):
            cost = _euclid(A[i - 1], B[j - 1])
            # min of insertion (i-1,j), match (i-1,j-1), deletion (i,j-1)
            choices = (D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
            arg = int(np.argmin(choices))
            best = choices[arg]
            D[i, j] = cost + best
            if arg == 0:   # came from (i-1, j)
                phi_i[i, j], phi_j[i, j] = i - 1, j
            elif arg == 1: # came from (i-1, j-1)
                phi_i[i, j], phi_j[i, j] = i - 1, j - 1
            else:          # came from (i, j-1)
                phi_i[i, j], phi_j[i, j] = i, j - 1

    # Backtrack path from (n, m)
    path: List[Tuple[int, int]] = []
    i, j = n, m
    if not np.isfinite(D[n, m]):
        # No feasible path (band too tight).
        return float("inf"), path
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))  # convert to 0-based frame indices
        pi, pj = phi_i[i, j], phi_j[i, j]
        i, j = pi, pj
    path.reverse()
    return float(D[n, m]), path


def _dtw_align(A: np.ndarray, B: np.ndarray, config: dict) -> Tuple[float, List[Tuple[int, int]], str]:
    """
    Align two sequences of shape [T, D] using DTW.
    Tries fastdtw if available (and allowed), else falls back to windowed DP.
    Returns (total_cost, path, method_used).
    """
    # Choose method
    use_fastdtw = bool(config.get("use_fastdtw", True))
    radius = int(config.get("dtw_radius", max(5, int(0.1 * max(A.shape[0], B.shape[0])))))  # in frames

    if use_fastdtw:
        try:
            from fastdtw import fastdtw
            total_cost, path = fastdtw(A, B, dist=_euclid)
            return float(total_cost), list(path), "fastdtw"
        except Exception as e:
            logger.warning("fastdtw unavailable or failed (%s); falling back to windowed DTW.", e)

    # Fall back to windowed DTW
    total_cost, path = _dtw_windowed(A, B, radius=radius)
    return total_cost, path, f"windowed(radius={radius})"


# ---------- Step 3 helpers   ----------
def _build_feature_names(include_velocities: bool) -> List[str]:
    """
    Order must match _frame_features() and _append_velocities().
    Base (10 dims): R_elbow, R_shoulder, R_hip, R_knee,
                    L_elbow, L_shoulder, L_hip, L_knee,
                    torso_tilt, coil
    If velocities included, append the same names with '_vel' suffix.
    """
    base = [
        "R_elbow", "R_shoulder", "R_hip", "R_knee",
        "L_elbow", "L_shoulder", "L_hip", "L_knee",
        "torso_tilt", "coil"
    ]
    if include_velocities:
        return base + [f"{n}_vel" for n in base]
    return base


def _standardize_pair(F_me: np.ndarray, F_pro: np.ndarray, method: str = "zscore") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Standardize features using joint stats from both sequences to ensure comparability.
    method: 'zscore' (mean/std) or 'minmax' (min/max). Returns standardized copies and stats.
    """
    X = np.vstack([F_me, F_pro]).astype(np.float32, copy=False)
    eps = 1e-6
    stats = {"method": method}

    if method.lower() == "minmax":
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        rng = np.maximum(xmax - xmin, eps)
        Fm = (F_me - xmin) / rng
        Fp = (F_pro - xmin) / rng
        stats.update({"min": xmin.tolist(), "max": xmax.tolist()})
    else:  # default zscore
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        Fm = (F_me - mu) / sd
        Fp = (F_pro - mu) / sd
        stats.update({"mean": mu.tolist(), "std": sd.tolist()})

    return Fm.astype(np.float32, copy=False), Fp.astype(np.float32, copy=False), stats


def _apply_feature_weights(F: np.ndarray, feature_names: List[str], weights_cfg: Dict[str, float]) -> Tuple[np.ndarray, List[float]]:
    """
    Multiply columns by weights based on a config dict like:
      {"coil": 1.5, "R_shoulder": 1.3, "torso_tilt_vel": 0.8, ...}
    Names that aren't in the config get weight 1.0.
    """
    if not isinstance(weights_cfg, dict) or F.size == 0:
        return F, [1.0] * (F.shape[1] if F.ndim == 2 else 0)

    w = np.ones((F.shape[1],), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for name, val in weights_cfg.items():
        i = name_to_idx.get(name)
        if i is not None:
            try:
                w[i] = float(val)
            except Exception:
                pass
    Fw = F * w  # broadcast by column
    return Fw, w.tolist()

# ---------- Geometry helpers ----------
def _pt(fr: Dict[int, Tuple[float, float, float, float]], idx: int, vmin: float = 0.2) -> Optional[Tuple[float, float]]:
    """Get (x,y) if joint present with sufficient visibility, else None."""
    if idx not in fr:
        return None
    x, y, _z, v = fr[idx]
    if v < vmin:
        return None
    return (x, y)

def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle ABC in degrees; returns NaN if any point is None."""
    if a is None or b is None or c is None:
        return float('nan')
    ax, ay = a; bx, by = b; cx, cy = c
    ab = np.array([ax - bx, ay - by], dtype=np.float32)
    cb = np.array([cx - bx, cy - by], dtype=np.float32)
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-6
    cosang = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))

def _line_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Angle of vector p1->p2 vs. horizontal (x+) in degrees, signed (-180..180)."""
    if p1 is None or p2 is None:
        return float('nan')
    dx, dy = (p2[0] - p1[0], p2[1] - p1[1])
    return float(math.degrees(math.atan2(dy, dx)))

def _mid(p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

def _torso_tilt_deg(fr: Dict[int, Tuple[float, float, float, float]]) -> float:
    """Angle between hip-mid -> shoulder-mid and vertical up; 0 = upright."""
    ls = _pt(fr, L_SHOULDER); rs = _pt(fr, R_SHOULDER)
    lh = _pt(fr, L_HIP);      rh = _pt(fr, R_HIP)
    s_mid = _mid(ls, rs); h_mid = _mid(lh, rh)
    if s_mid is None or h_mid is None:
        return float('nan')
    # vector from hips to shoulders
    vx, vy = (s_mid[0] - h_mid[0], s_mid[1] - h_mid[1])
    # vertical up in our normalized space is (0, -1)
    denom = (math.hypot(vx, vy) + 1e-6)
    cosang = (-(vy) / denom)  # dot((vx,vy), (0,-1)) = -vy
    cosang = max(-1.0, min(1.0, cosang))
    deg = math.degrees(math.acos(cosang))
    # sign by horizontal direction (optional; we can keep unsigned for now)
    return float(deg)

def _coil_deg(fr: Dict[int, Tuple[float, float, float, float]]) -> float:
    """Hip–shoulder separation (absolute difference between their line angles)."""
    ls = _pt(fr, L_SHOULDER); rs = _pt(fr, R_SHOULDER)
    lh = _pt(fr, L_HIP);      rh = _pt(fr, R_HIP)
    if ls is None or rs is None or lh is None or rh is None:
        return float('nan')
    theta_sh = _line_angle(ls, rs)   # shoulder line angle
    theta_hp = _line_angle(lh, rh)   # hip line angle
    # wrap diff into [0,180]
    diff = abs(theta_sh - theta_hp)
    while diff > 180.0:
        diff -= 360.0
    return float(abs(diff))

def _frame_features(fr: Dict[int, Tuple[float, float, float, float]]) -> List[float]:
    """Feature vector for one frame (angles in degrees)."""
    # Right side
    rs = _pt(fr, R_SHOULDER); re = _pt(fr, R_ELBOW); rw = _pt(fr, R_WRIST)
    rh = _pt(fr, R_HIP);      rk = _pt(fr, R_KNEE);  ra = _pt(fr, R_ANKLE)
    right_elbow   = _angle(rs, re, rw)      # elbow flex
    right_shoulder= _angle(rh, rs, re)      # hip-shoulder-elbow chain
    right_hip     = _angle(rs, rh, rk)      # shoulder-hip-knee
    right_knee    = _angle(rh, rk, ra)      # hip-knee-ankle

    # Left side
    ls = _pt(fr, L_SHOULDER); le = _pt(fr, L_ELBOW); lw = _pt(fr, L_WRIST)
    lh = _pt(fr, L_HIP);      lk = _pt(fr, L_KNEE);  la = _pt(fr, L_ANKLE)
    left_elbow    = _angle(ls, le, lw)
    left_shoulder = _angle(lh, ls, le)
    left_hip      = _angle(ls, lh, lk)
    left_knee     = _angle(lh, lk, la)

    # Global features
    torso_tilt    = _torso_tilt_deg(fr)
    coil          = _coil_deg(fr)

    # You can add more later (e.g., wrist relative to head, ankle dorsiflexion, etc.)
    # For robustness, we keep NaNs and will handle them (drop or impute) before DTW.
    return [
        right_elbow, right_shoulder, right_hip, right_knee,
        left_elbow,  left_shoulder,  left_hip,  left_knee,
        torso_tilt,  coil
    ]

def _sequence_features(frames: List[Dict[int, Tuple[float, float, float, float]]]) -> np.ndarray:
    """Convert a list of frame dicts into a 2D array [T, D] of features."""
    feats = [ _frame_features(fr) for fr in frames ]
    F = np.array(feats, dtype=np.float32)  # shape [T, D]
    return F

def _append_velocities(F: np.ndarray, fps: int) -> np.ndarray:
    """Optionally append per-feature velocities (first difference * fps) to the feature matrix."""
    if F.size == 0 or F.shape[0] < 2:
        return F
    # Finite differences along time axis; keep same length by padding first row with zeros
    dF = np.diff(F, axis=0, prepend=F[0:1, :]) * float(fps)
    return np.concatenate([F, dF], axis=1)


def calculate_movement_similarity(me_norm_path: Path, pro_norm_path: Path, fps: int, config: dict) -> dict:
    """
    Compare two normalized landmark sequences (me vs pro) and produce
    a similarity score representing how similar the motions are.
    """
    # 1. Load and parse both *_norm.json files.
    #     - Extract the list of frames and per-frame landmark coordinates.
    #     - Ensure both contain enough valid frames for comparison.
    def _load_norm_landmarks(path: Path) -> List[Dict[int, Tuple[float, float, float, float]]]:
        """Read a normalized landmarks JSON and return list of frame dicts {idx: (x,y,z,v)}."""
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Normalized landmarks file not found: {p}")
        data = json.loads(p.read_text())
        if "frames" not in data:
            raise ValueError(f"Malformed JSON: missing 'frames' in {p}")

        frames = []
        for frame in data["frames"]:
            lm = frame.get("lm", {})
            parsed = {}
            for k, arr in lm.items():
                try:
                    idx = int(k)
                    if not isinstance(arr, (list, tuple)) or len(arr) < 4:
                        continue
                    x, y, z, v = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
                    parsed[idx] = (x, y, z, v)
                except Exception:
                    continue
            frames.append(parsed)
        return frames

    logger.info("Loading normalized landmark data for comparison…")
    me_frames = _load_norm_landmarks(me_norm_path)
    pro_frames = _load_norm_landmarks(pro_norm_path)
    logger.info("Loaded %d frames for me, %d frames for pro.", len(me_frames), len(pro_frames))

    # Sanity check: ensure we have enough data for comparison
    min_frames_required = config.get("min_frames_required", 10)
    if len(me_frames) < min_frames_required or len(pro_frames) < min_frames_required:
        logger.warning("Insufficient frames for comparison (me=%d, pro=%d, min=%d).",
                       len(me_frames), len(pro_frames), min_frames_required)
        return {
            "score": None,
            "avg_dist": None,
            "path_len": None,
            "per_phase": {},
            "feature_contrib": {},
            "warning": "Insufficient frames for comparison"
        }

    # 2. Compute a consistent feature vector for each frame.
    #     - For each frame, calculate key pose-based features:
    #         a) Joint angles (elbow, shoulder, hip, knee, ankle)
    #         b) Torso tilt, hip-shoulder separation (coil)
    #         c) Optional angular velocities (Δangle per frame)
    #     - This produces sequences like: [frame_0_features, frame_1_features, …]
    #       for both players.
    
    #       2a) raw feature matrices [T, D]
    F_me  = _sequence_features(me_frames)   # degrees for angles, degrees for tilt/coil
    F_pro = _sequence_features(pro_frames)

    #       2b) handle NaNs (simple forward fill, then zero-fill as fallback)
    def _clean_nan(F: np.ndarray) -> np.ndarray:
        if F.size == 0:
            return F
        # forward fill along time
        for t in range(1, F.shape[0]):
            nan_mask = np.isnan(F[t])
            if np.any(nan_mask):
                F[t, nan_mask] = F[t-1, nan_mask]
        # if first row still has NaNs, set to column means (ignoring NaNs), then zeros if necessary
        col_means = np.nanmean(F, axis=0)
        inds = np.where(np.isnan(F))
        F[inds] = np.take(col_means, inds[1])
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        return F

    F_me  = _clean_nan(F_me)
    F_pro = _clean_nan(F_pro)

    # 2c) optional: append angular velocities (Δ per second)
    include_vel = bool(config.get("include_velocities", True))   # <-- ADD THIS LINE
    if include_vel:
        F_me  = _append_velocities(F_me, fps=fps)
        F_pro = _append_velocities(F_pro, fps=fps)

    logger.info("Feature shapes: me=%s, pro=%s (T×D)", F_me.shape, F_pro.shape)

    # 3. Normalize / standardize features.
    #     - Standardize each feature dimension (e.g., z-score or min-max)
    #       so that distances are comparable across joints and angles.
    #     - Apply any feature weighting defined in config (e.g., more weight to upper-body features).
    
    #       3a) choose method
    method = str(config.get("standardize", "zscore")).lower()  # 'zscore' or 'minmax'

    #       3b) standardize BOTH sequences using JOINT stats (ensures same scale)
    F_me_std, F_pro_std, stats = _standardize_pair(F_me, F_pro, method=method)

    #       3c) apply feature weights (by name)
    feature_names = _build_feature_names(include_velocities=include_vel)
    weights_cfg = config.get("feature_weights", {
        # sensible defaults (tune later):
        "coil": 1.3,
        "R_shoulder": 1.2, "L_shoulder": 1.1,
        "torso_tilt": 1.0,
        # velocities often informative; sample defaults:
        "coil_vel": 1.1, "R_shoulder_vel": 1.1, "L_shoulder_vel": 1.05
    })

    F_me_w, weights_vec = _apply_feature_weights(F_me_std, feature_names, weights_cfg)
    # use the SAME weights on pro (same feature ordering)
    F_pro_w, _ = _apply_feature_weights(F_pro_std, feature_names, weights_cfg)

    logger.info("Features standardized via %s and weights applied (D=%d).", method, F_me_w.shape[1])

    # 4. Align the two sequences in time.
    #     - Use Dynamic Time Warping (DTW) or a similar alignment algorithm
    #       to handle small differences in serve timing or tempo.
    #     - DTW will output an alignment path between frames of me vs pro.

    # Use either fastdtw (if allowed/available) or our windowed DP with a Sakoe–Chiba band.
    total_cost, path, method_used = _dtw_align(F_me_w, F_pro_w, config)

    if not path:
        logger.warning("DTW produced no feasible path (method=%s).", method_used)
        return {
            "score": None,
            "avg_dist": None,
            "path_len": 0,
            "per_phase": {},
            "feature_contrib": {},
            "dtw": {"method": method_used, "total_cost": float(total_cost)},
            "frames_me": len(me_frames),
            "frames_pro": len(pro_frames),
            "feature_dim": int(F_me_w.shape[1]) if F_me_w.ndim == 2 else 0,
            "T_me": int(F_me_w.shape[0]) if F_me_w.ndim == 2 else 0,
            "T_pro": int(F_pro_w.shape[0]) if F_pro_w.ndim == 2 else 0,
            "standardization": {"method": stats.get("method")},
            "weights_used": {k: float(v) for k, v in weights_cfg.items()},
            "warning": "No feasible DTW path (band too tight or empty sequences)."
        }

    # Compute per-step distances along the alignment path on the standardized+weighted features.
    # This yields an interpretable average distance independent of path length.
    step_dists = [
        _euclid(F_me_w[i], F_pro_w[j]) for (i, j) in path
        if 0 <= i < F_me_w.shape[0] and 0 <= j < F_pro_w.shape[0]
    ]
    avg_dist = float(np.mean(step_dists)) if step_dists else float("inf")
    path_len = len(path)

    logger.info("DTW alignment done (method=%s, path_len=%d, avg_step_dist=%.4f)",
                method_used, path_len, avg_dist)

    # 5. Compute framewise distances along the DTW path.
    #     - For each matched pair of frames, compute Euclidean distance between
    #       feature vectors (weighted if applicable).
    #     - Average these distances to get an overall mean motion difference.
    dist_metric = str(config.get("distance_metric", "euclidean"))
    trim_fraction = float(config.get("distance_trim_fraction", 0.0))  # e.g., 0.05 to trim 5% tails
    step_dists, avg_dist = _path_distances(F_me_w, F_pro_w, path,
                                           metric=dist_metric,
                                           trim_fraction=trim_fraction)
    path_len = len(path)
    logger.info("Computed DTW path distances (path_len=%d, avg_step_dist=%.4f, metric=%s, trim=%.2f)",
                path_len, avg_dist, dist_metric, trim_fraction)

    # 6. Convert average distance to similarity score.
    #     - Map the average feature distance to a 0-100 similarity scale.
    #       Example: score = 100 * exp(-o * avg_dist)
    #     - Optionally clip/normalize to stay within [0, 100].
    # Map avg_dist (larger = more different) to a bounded 0–100 similarity.
    # Formula: score = 100 * exp(-alpha * avg_dist)
    # Alpha controls how quickly similarity decays as distance grows.
    alpha = float(config.get("similarity_alpha", 3.0))
    max_clip = float(config.get("similarity_max_clip", 100.0))
    min_clip = float(config.get("similarity_min_clip", 0.0))

    if np.isfinite(avg_dist) and avg_dist >= 0.0:
        score = 100.0 * math.exp(-alpha * avg_dist)
    else:
        score = 0.0

    # optional scaling so that "perfect" alignment scores near 100, and large distances flatten near 0
    score = max(min_clip, min(max_clip, score))

    logger.info("Similarity score computed: %.2f (alpha=%.2f, avg_dist=%.4f)",
                score, alpha, avg_dist)
    
    # 7. (Optional) Phase analysis.
    #     - Roughly segment each serve into phases (load, trophy, drop, acceleration, contact, follow-through).
    #     - Compute per-phase DTW distances and store as sub-scores in "per_phase".

    # 8. (Optional) Feature contribution analysis.
    #     - Compute how much each feature dimension contributed to the total DTW cost.
    #     - This can later be used to explain *where* two motions differ (e.g., hip vs shoulder).

    # 9. Assemble the results into a dictionary.
    #     return {
    #         "score": <float>,              # overall 0-100 similarity
    #         "avg_dist": <float>,           # mean distance between aligned frames
    #         "path_len": <int>,             # DTW path length
    #         "per_phase": {...},            # optional breakdown by serve phase
    #         "feature_contrib": {...}       # optional breakdown by joint/angle
    #     }
    # ---- Step 9: assemble results ----
    result = {
        "score": float(score),                # overall 0–100 similarity
        "avg_dist": float(avg_dist),          # mean distance over DTW path (on std+weighted features)
        "path_len": int(path_len),            # number of matched pairs
        "per_phase": {},                      # (skipped for now; Step 7)
        "feature_contrib": {},                # (skipped for now; Step 8)
        "dtw": {
            "method": str(method_used),
            "total_cost": float(total_cost),
            # keep a small sample so results aren’t huge
            "path_sample": path[::max(1, path_len // 10)] if path_len > 0 else []
        },
        "frames_me": int(len(me_frames)),
        "frames_pro": int(len(pro_frames)),
        "feature_dim": int(F_me_w.shape[1]) if F_me_w.ndim == 2 else 0,
        "T_me": int(F_me_w.shape[0]) if F_me_w.ndim == 2 else 0,
        "T_pro": int(F_pro_w.shape[0]) if F_pro_w.ndim == 2 else 0,
        "standardization": {"method": stats.get("method")},
        "weights_used": {k: float(v) for k, v in weights_cfg.items()},
        "config": {
            "include_velocities": bool(include_vel),
            "distance_metric": str(config.get("distance_metric", "euclidean")),
            "distance_trim_fraction": float(config.get("distance_trim_fraction", 0.0)),
            "similarity_alpha": float(config.get("similarity_alpha", 3.0)),
            "standardize": str(config.get("standardize", "zscore")),
            "dtw_radius": int(config.get("dtw_radius", max(5, int(0.1 * max(F_me_w.shape[0], F_pro_w.shape[0]))))),
            "use_fastdtw": bool(config.get("use_fastdtw", True)),
        },
    }

    # ---- Step 10: log summary ----
    logger.info(
        "Movement similarity: %.1f (avg_dist=%.4f, path_len=%d, method=%s)",
        result["score"], result["avg_dist"], result["path_len"], result["dtw"]["method"]
    )

    return result
