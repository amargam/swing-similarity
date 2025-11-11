✅ current accomplishments

CLI + logging

command_line_parser.py with clean argparse groups, --debug, --overwrite, etc.

main.py orchestration with proper logging config, timestamped runs/<ts>/{inputs,outputs} layout, meta.json.

Pose extraction (foundational)

pose_extractor.py: OpenCV + MediaPipe Pose.

Per-frame raw and normalized landmarks (pelvis-origin, torso-scaled).

Global orientation lock (no more “flippy” twin); honors lefty flags.

Light EMA smoothing for stability.

Outputs: landmarks_*_raw.json, landmarks_*_norm.json.

Digital twin rendering

twin_renderer.py: black background, joints + connections, auto-fit scale.

Twins for both user + pro; handedness fixed so the twin matches the raw video’s left/right.

Movement similarity (v1)

calculations.py:

Step 1–3: load normalized landmarks → compute per-frame feature vectors (joint angles, torso tilt, hip–shoulder “coil”) → standardize and weight features; optional velocities.

Step 4–6: DTW alignment (fastdtw or windowed band) → path distances (robust trimmed mean) → 0–100 score via exponential map.

Step 9–10: return structured result + log summary (score, avg_dist, path_len, method).

main.py now runs similarity and writes a JSON summary.

Artifacts & reproducibility

Every run keeps: copied inputs, twin videos, landmark dumps, meta.json, similarity output.

Predictable paths for inspection and future training.

🎯 goal we’re adding next

Racket keypoints via a learned model (robust):

Two new landmarks per frame:

racket_grip (butt cap / base of the handle)

racket_tip (topmost tip)

Treat as first-class landmarks in extraction, normalization, rendering, and later analytics.

🗺️ roadmap (incremental, practical)
phase A — data & schema

Schema extension

In pose_extractor.py normalized payload, add keys for racket_grip and racket_tip with [x, y, z, visibility].

Reserve stable indices for convenience (e.g., 100=racket_grip, 101=racket_tip) or store under named keys; renderer supports both.

Renderer update

twin_renderer.py: if racket landmarks exist, draw line grip↔tip with a distinct color; draw small nodes at each end.

Config flags (no-op until model is wired)

Parser: --track-racket, --racket-model path/to/weights, --racket-conf 0.4.

phase B — model selection & training pipeline

Model choice (2-keypoint pose)

Use a lightweight keypoint detector:

Option A: YOLOv8/YOLOv9 pose fine-tuned with 2 keypoints (grip, tip).

Option B: RTMPose (2-keypoint head).

Export to ONNX for easy inference (onnxruntime) or keep .pt with PyTorch.

Dataset prep

Collect frames from your runs (both you + pro). Aim for diverse lighting/cameras.

Label grip and tip; store in COCO keypoint format (2 keypoints).

Split train/val; maintain a small test subset from unseen sessions.

Training script

Minimal fine-tune (frozen backbone OK to start).

Augmentations that help:

motion blur, brightness/contrast, small rotations, random crops around hands.

Track mAP@OKS or PCK for 2 keypoints; save best checkpoint.

phase C — inference integration

racket_tracker.py

Add infer_racket_keypoints(frame_bgr, model, conf) → {grip:(x,y,conf), tip:(x,y,conf)}

Optional fusion with wrist prior: bias search ROI near the racket-hand wrist from pose (fewer false positives).

Plug into pose_extractor.py

Load model lazily if --track-racket is on.

For each frame:

run inference (full-frame or ROI)

temporal smoothing (EMA or small Kalman)

normalize to pelvis/torsoscale with fixed sign, just like joints

write to raw & normalized JSON.

Validate on your existing videos

Inspect twin videos with racket drawn; check stability of racket drop and contact frames.

phase D — use racket for analytics (later)

Movement similarity add-ons

Append racket angle (grip→tip vs forearm), tip height, and tip velocity to feature vectors.

Weight these moderately in movement similarity (they matter a lot visually).

Efficiency metrics v1

Use peak timing and velocity ratios including tip speed:

hip rot vel → trunk vel → shoulder/elbow → wrist→tip.

Racket drop depth and lag timing become strong proxies for loading/transfer.

Explainability

Report where sequences diverge most (e.g., “tip lag peaks 60 ms later; coil 8° lower at load”).

phase E — engineering polish

Config + packaging

YAML/JSON config for weights and thresholds; --config path.

Package as a CLI with pyproject.toml and entry point serve-compare.

Testing

Unit tests for geometry, DTW, standardization, I/O schemas.

Golden tests on a tiny sample to catch regressions.

Performance

Batch inference or frame skipping when source FPS is high.

Optional GPU path for model inference; fallback CPU.

🔧 current module map (target)
serve_compare/
  main.py
  command_line_parser.py
  pose_extractor.py
  twin_renderer.py
  calculations.py
  racket_tracker.py           # (new) learned keypoint inference + smoothing
  runs/                       # artifacts by timestamp

📌 tuning notes to remember

Keep the global facing sign fixed for each clip (done).

When adding racket points, normalize them exactly like joints (pelvis-origin, torso-scale, same sign).

Start similarity with angles + coil + velocities, then sprinkle racket features once stable.

For efficiency metrics, rely on peak timings and ratios rather than absolute speeds (more robust to camera/scale).