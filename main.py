from __future__ import annotations
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import shutil
import platform

from command_line_parser import build_parser, validate_outputs




# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # ensure VS Code/others don't block our config
    )

logger = logging.getLogger(__name__)



# ──────────────────────────────────────────────────────────────────────────────
# Run folder + metadata
# ──────────────────────────────────────────────────────────────────────────────

def make_run_dir(base: Path = Path("runs")) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (base / ts).resolve()
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)
    return run_dir

def copy_inputs(run_dir: Path, me: Path, pro: Path) -> tuple[Path, Path]:
    inputs_dir = run_dir / "inputs"
    me_copy = inputs_dir / me.name
    pro_copy = inputs_dir / pro.name
    if me_copy.exists():
        logger.debug("Input already present: %s", me_copy)
    else:
        shutil.copy2(me, me_copy)
        logger.debug("Copied input -> %s", me_copy)
    if pro_copy.exists():
        logger.debug("Input already present: %s", pro_copy)
    else:
        shutil.copy2(pro, pro_copy)
        logger.debug("Copied input -> %s", pro_copy)
    return me_copy, pro_copy

def write_meta(run_dir: Path, args, extras: dict | None = None) -> Path:
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool_version": "0.1.0",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "args": {
            "me": str(args.me),
            "pro": str(args.pro),
            "out_video": str(args.out_video),
            "out_json": str(args.out_json),
            "fps": args.fps,
            "ghost": args.ghost,
            "side_by_side": args.side_by_side,
            "lefty_me": args.lefty_me,
            "lefty_pro": args.lefty_pro,
            "use_3d": args.use_3d,
            "overwrite": args.overwrite,
            "debug": args.debug,
        },
    }
    if extras:
        meta.update(extras)
    meta_path = run_dir / "outputs" / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote meta.json -> %s", meta_path)
    return meta_path


# region PIPELINE

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stage interfaces (stubs for now)
# ──────────────────────────────────────────────────────────────────────────────
# We’ll create these modules next; for now, the functions are defined inline as
# placeholders so the structure is clear and runnable.

from pose_extractor import extract_and_normalize_pose
from twin_renderer import render_twin_2d

def save_raw_copy(src_video: Path, out_path: Path) -> None:
    """
    For UX consistency keep a raw alongside outputs (copy or re-encode later).
    """
    if out_path.exists():
        logger.debug("Raw already exists: %s", out_path)
        return
    
    shutil.copy2(src_video, out_path)
    logger.info("Saved raw copy -> %s", out_path)

from calculations import calculate_movement_similarity

# endregion PIPELINE
# region MAIN

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    # 1) Parse args (argparse handles --help and exits before here if requested)
    parser = build_parser()
    args = parser.parse_args(argv)

    # 2) Configure logging
    configure_logging(args.debug)
    logger.info("Starting serve comparison tool…")

    # 3) Validate/prepare output file paths (global “final” outputs for CLI)
    out_video, out_json = validate_outputs(args)

    # 4) Create run folder and record everything
    run_dir = make_run_dir()
    me_in, pro_in = copy_inputs(run_dir, args.me, args.pro)
    write_meta(run_dir, args)

    # 5) Stage A: pose extraction + normalization → landmarks dumps
    me_pose = extract_and_normalize_pose(me_in, fps=args.fps, use_3d=args.use_3d, lefty=args.lefty_me)
    pro_pose = extract_and_normalize_pose(pro_in, fps=args.fps, use_3d=args.use_3d, lefty=args.lefty_pro)

    # 6) Stage B: render twins (2D) and save raws
    outputs_dir = run_dir / "outputs"
    me_raw_out = outputs_dir / f"{me_in.stem}_raw.mp4"
    pro_raw_out = outputs_dir / f"{pro_in.stem}_raw.mp4"
    me_twin_out = outputs_dir / f"{me_in.stem}_twin2d.mp4"
    pro_twin_out = outputs_dir / f"{pro_in.stem}_twin2d.mp4"

    save_raw_copy(me_in, me_raw_out)
    save_raw_copy(pro_in, pro_raw_out)

    render_twin_2d(Path(me_pose["landmarks_norm_path"]), me_twin_out, args.fps)
    render_twin_2d(Path(pro_pose["landmarks_norm_path"]), pro_twin_out, args.fps)

    # 7) (Later) Stage C: movement similarity + efficiency metrics
    #    - align_and_score(...)
    #    - efficiency(...)
    #    - write JSON summaries and a combined top-level out_json
    # Build a small config you can tweak without code changes
    ms_config = {
        "min_frames_required": 10,
        "include_velocities": True,
        "standardize": "zscore",              # or "minmax"
        "feature_weights": {                  # tune later
            "coil": 1.3,
            "R_shoulder": 1.2, "L_shoulder": 1.1,
            "torso_tilt": 1.0,
            "coil_vel": 1.1, "R_shoulder_vel": 1.1, "L_shoulder_vel": 1.05,
        },
        "use_fastdtw": True,
        "dtw_radius": 15,                     # frames; used if fastdtw unavailable
        "distance_metric": "euclidean",
        "distance_trim_fraction": 0.05,       # trim 5% tails for robustness
        "similarity_alpha": 3.0,              # score = 100*exp(-alpha*avg_dist)
    }

    ms_result = calculate_movement_similarity(
        Path(me_pose["landmarks_norm_path"]),
        Path(pro_pose["landmarks_norm_path"]),
        fps=args.fps,
        config=ms_config,
    )

    # Print to console (via logger) so you immediately see it
    if ms_result.get("score") is not None:
        logger.info("Movement similarity score: %.1f", ms_result["score"])
    else:
        logger.warning("Movement similarity could not be computed (avg_dist=%s, path_len=%s)",
                       ms_result.get("avg_dist"), ms_result.get("path_len"))

    # Write JSON summary to the user-specified out_json path
    summary_payload = {
        "movement_similarity": ms_result,
        # You can add more sections here later:
        # "efficiency_me": {},
        # "efficiency_pro": {},
        # "efficiency_similarity": {},
    }
    (outputs_dir / Path(args.out_json).name).write_text(json.dumps(summary_payload, indent=2))
    logger.info("Wrote similarity summary -> %s", outputs_dir / Path(args.out_json).name)

    logger.info("Primary goals completed for this run (raw + twin videos + similarity).")
    logger.info("Artifacts in: %s", outputs_dir)

if __name__ == "__main__":
    sys.exit(main())

# endregion
