# main.py
"""
End-to-end driver for the semantic driving world model.

Pipeline:
  [0] extract_frames          → data/frames/{clip}/
  [1] run_vlm_inference       → data/predictions/world_state_claude.csv
  [2] temporal_smoothing      → data/predictions/world_state_claude_smoothed.csv
  [3] segment_world_state     → data/predictions/world_state_segments.csv
  [4] generate_segment_gloss  → data/predictions/segment_gloss.csv
  [5] build_state_and_planner → data/predictions/planner_commands.csv
  [6] overlay_world_state     → results/{clip}_overlay.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# We import overlay_world_state as a library to generate per-clip videos.
from src import overlay_world_state as ow


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
PRED_DIR = DATA_DIR / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default FPS per clip for nice playback
DEFAULT_FPS: Dict[str, int] = {
    "clip1": 2,
    "clip2": 3,
    "clip3": 3,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(label: str, cmd: List[str], skip: bool = False, force: bool = False) -> bool:
    """
    Run a subprocess command with nice logging.
    Returns True on success, False on failure.
    """
    if skip and not force:
        log(f"[↺] {label}: cached, skipping")
        return True

    log(f"[▶] {label}...")
    try:
        subprocess.run(cmd, check=True)
        log(f"[✓] {label}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"[✖] {label} FAILED with exit code {e.returncode}")
        return False


def frames_exist_for_clip(clip: str) -> bool:
    d = FRAMES_DIR / clip
    if not d.exists():
        return False
    jpgs = list(d.glob("frame_*.jpg"))
    pngs = list(d.glob("frame_*.png"))
    return len(jpgs) > 0 or len(pngs) > 0


def ensure_frames(clips: List[str], force: bool = False) -> bool:
    """
    Ensure frames exist for the requested clips.
    Currently extract_frames.py is global, so we call it once if any clip is missing.
    """
    need_extract = force
    if not need_extract:
        for c in clips:
            if not frames_exist_for_clip(c):
                need_extract = True
                break

    cmd = [sys.executable, "-m", "src.extract_frames"]
    return run_cmd("extract_frames", cmd, skip=not need_extract, force=force)


def main():
    parser = argparse.ArgumentParser(description="Run the driving world model pipeline end-to-end.")
    parser.add_argument(
        "--clip",
        type=str,
        help="Clip name to process (e.g. clip1, clip2, clip3). If omitted and --all is not set, defaults to clip1.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all clips: clip1, clip2, clip3.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="FPS for overlay video (if omitted, uses clip-specific defaults).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all stages (ignore caches).",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Same as --force, but conceptually for all clips (alias).",
    )

    args = parser.parse_args()

    # Determine which clips to run
    if args.all:
        # Automatically detect all clips from frames directory
        if FRAMES_DIR.exists():
            clips = sorted([d.name for d in FRAMES_DIR.iterdir() if d.is_dir()])
            if not clips:
                log("No clips found in frames directory. Run extract_frames first.")
                sys.exit(1)
        else:
            # Fallback to default clips if frames directory doesn't exist yet
            clips = ["clip1", "clip2", "clip3"]
    elif args.clip:
        clips = [args.clip]
    else:
        clips = ["clip1"]  # default

    force = bool(args.force or args.force_all)

    log("====================================")
    log("  Driving World Model Pipeline")
    log("====================================")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Clips: {', '.join(clips)}")
    log(f"Force recompute: {force}")
    if args.fps:
        log(f"Global FPS override: {args.fps}")
    log("")

    # -------------------------------------------------------
    # [0] Ensure frames exist for requested clips
    # -------------------------------------------------------
    if not ensure_frames(clips, force=force):
        log("Aborting: frame extraction failed.")
        sys.exit(1)

    # -------------------------------------------------------
    # [1] VLM inference (global CSV)
    # -------------------------------------------------------
    world_state_csv = PRED_DIR / "world_state_claude.csv"
    vlm_skip = world_state_csv.exists() and not force
    if vlm_skip:
        log(f"[↺] world_state_claude.csv already exists at {world_state_csv}")
    if not run_cmd(
        "run_vlm_inference",
        [sys.executable, "-m", "src.run_vlm_inference"],
        skip=vlm_skip,
        force=force,
    ):
        log("Aborting: VLM inference failed.")
        sys.exit(1)

    # -------------------------------------------------------
    # [2] Temporal smoothing
    # -------------------------------------------------------
    smoothed_csv = PRED_DIR / "world_state_claude_smoothed.csv"
    smoothing_skip = smoothed_csv.exists() and not force
    if smoothing_skip:
        log(f"[↺] world_state_claude_smoothed.csv already exists at {smoothed_csv}")
    if not run_cmd(
        "temporal_smoothing",
        [sys.executable, "-m", "src.temporal_smoothing"],
        skip=smoothing_skip,
        force=force,
    ):
        log("Aborting: temporal smoothing failed.")
        sys.exit(1)

    # -------------------------------------------------------
    # [3] Segment world state
    # -------------------------------------------------------
    segments_csv = PRED_DIR / "world_state_segments.csv"
    segment_skip = segments_csv.exists() and not force
    if segment_skip:
        log(f"[↺] world_state_segments.csv already exists at {segments_csv}")
    if not run_cmd(
        "segment_world_state",
        [sys.executable, "-m", "src.segment_world_state"],
        skip=segment_skip,
        force=force,
    ):
        log("Aborting: segmentation failed.")
        sys.exit(1)

    # -------------------------------------------------------
    # [4] Generate segment gloss
    # -------------------------------------------------------
    gloss_csv = PRED_DIR / "segment_gloss.csv"
    gloss_skip = gloss_csv.exists() and not force
    if gloss_skip:
        log(f"[↺] segment_gloss.csv already exists at {gloss_csv}")
    if not run_cmd(
        "generate_segment_gloss",
        [sys.executable, "-m", "src.generate_segment_gloss"],
        skip=gloss_skip,
        force=force,
    ):
        log("Aborting: gloss generation failed.")
        sys.exit(1)

    # -------------------------------------------------------
    # [5] Build state machine + planner commands
    # -------------------------------------------------------
    planner_csv = PRED_DIR / "planner_commands.csv"
    planner_skip = planner_csv.exists() and not force
    if planner_skip:
        log(f"[↺] planner_commands.csv already exists at {planner_csv}")
    if not run_cmd(
        "build_state_and_planner",
        [sys.executable, "-m", "src.build_state_and_planner"],
        skip=planner_skip,
        force=force,
    ):
        log("Aborting: planner build failed.")
        sys.exit(1)

    # -------------------------------------------------------
    # [6] Overlay videos per clip (call overlay_world_state as a library)
    # -------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions / segments / gloss / planner once
    preds = ow.load_smoothed_preds(smoothed_csv)
    segments = ow.load_segments(segments_csv)
    glosses = ow.load_gloss(gloss_csv)
    planner = ow.load_planner(planner_csv)

    overlay_failures = []

    for clip in clips:
        fps = args.fps if args.fps is not None else DEFAULT_FPS.get(clip, 2)
        out_path = RESULTS_DIR / f"{clip}_overlay.mp4"

        if out_path.exists() and not force:
            log(f"[↺] overlay for {clip} already exists at {out_path}")
            continue

        log(f"[▶] overlay_world_state for {clip} (fps={fps})...")
        try:
            ow.make_overlay_video_for_clip(
                clip_name=clip,
                frames_root=FRAMES_DIR,
                preds=preds,
                segments=segments,
                gloss=glosses,
                planner=planner,
                out_dir=RESULTS_DIR,
                fps=fps,
            )
            log(f"[✓] overlay for {clip} → {out_path}")
        except Exception as e:
            log(f"[✖] overlay for {clip} FAILED: {e}")
            overlay_failures.append(clip)

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    log("\n====================================")
    log("  Pipeline complete")
    log("====================================")
    for clip in clips:
        out_path = RESULTS_DIR / f"{clip}_overlay.mp4"
        status = "FAILED" if clip in overlay_failures else ("OK" if out_path.exists() else "SKIPPED")
        log(f"{clip}: overlay status = {status}, file = {out_path if out_path.exists() else '—'}")

    if overlay_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
