"""
Command-line argument parsing for swing → 3D scene pipeline.

This module ONLY parses & validates CLI inputs. No CV logic here.
"""

from __future__ import annotations
import argparse
import pathlib
import logging
from typing import Tuple

# Inherits config from main.py
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────────────────────────────────────

def _existing_file(path_str: str) -> pathlib.Path:
    p = pathlib.Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {p}")
    return p


def _positive_int(v: str) -> int:
    try:
        x = int(v)
    except Exception:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {v}")
    if x <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return x


def _sport_choice(v: str) -> str:
    allowed = {"tennis", "golf", "baseball", "throw", "javelin", "generic"}
    val = v.lower()
    if val not in allowed:
        raise argparse.ArgumentTypeError(f"--sport must be one of {sorted(allowed)}")
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swing3d",
        description="Extract a normalized 3D skeleton scene from a single MP4."
    )

    io = parser.add_argument_group("inputs/outputs")
    io.add_argument(
        "--video", required=True, type=_existing_file,
        help="Path to the input swing video (MP4)."
    )
    io.add_argument(
        "--out-dir", default=None,
        help="Directory to write outputs. If omitted, a timestamped runs/<ts> folder is created."
    )

    proc = parser.add_argument_group("processing")
    proc.add_argument(
        "--fps-target", type=_positive_int, default=60,
        help="Target FPS for processing/rendering (default: 60)."
    )
    proc.add_argument(
        "--sport", type=_sport_choice, default="tennis",
        help="Sport preset to tune heuristics (default: tennis)."
    )
    proc.add_argument(
        "--hands", action="store_true", default=False,
        help="Enable hand/finger landmark extraction (21 joints per hand)."
    )

    # Placeholders for future iterations; safe to parse now, ignore if unused.
    adv = parser.add_argument_group("advanced (future-proof)")
    adv.add_argument(
        "--marker-mode", choices=["none", "aruco", "learned_kp"], default="none",
        help="Tool endpoint detection strategy (butt/tip). v0.1 may ignore."
    )
    adv.add_argument(
        "--scene-hints", default=None,
        help="Optional YAML with scene/player hints (target line, lefty, known lengths)."
    )
    adv.add_argument(
        "--use-metric", action="store_true", default=False,
        help="Attempt single-view metric scaling if hints/markers available."
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--debug", "-d", action="store_true", default=False,
        help="Enable DEBUG logging."
    )
    misc.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite files inside --out-dir if they already exist."
    )

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Optional helpers (kept minimal; main.py owns run-dir creation)
# ──────────────────────────────────────────────────────────────────────────────

def validate_inputs(args: argparse.Namespace) -> pathlib.Path:
    """
    Validate that the video exists and return its resolved path.
    (main.py will create out-dir, write artifacts, etc.)
    """
    video = pathlib.Path(args.video).expanduser().resolve()
    if not video.exists() or not video.is_file():
        raise SystemExit(f"--video not found: {video}")
    if video.suffix.lower() != ".mp4":
        logger.warning(f"Input does not have .mp4 extension: {video.name}")
    return video


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    ns = parser.parse_args()
    vp = validate_inputs(ns)
    logger.info("[OK] Parsed arguments:")
    logger.info(f"  video:      {vp}")
    logger.info(f"  out_dir:    {ns.out_dir or '(auto runs/<ts>)'}")
    logger.info(f"  fps_target: {ns.fps_target}, sport={ns.sport}, hands={ns.hands}")
    logger.info(f"  marker_mode={ns.marker_mode}, scene_hints={ns.scene_hints}, use_metric={ns.use_metric}")
    logger.info(f"  debug={ns.debug}, overwrite={ns.overwrite}")
    sys.exit(0)
