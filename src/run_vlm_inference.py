# src/run_vlm_inference.py

import csv
from pathlib import Path
from typing import List

from .vlm_infer import VLMWorldModel


def list_frames(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in dir_path.iterdir() if p.suffix.lower() in exts]
    return sorted(files)


def main():
    project_root = Path(__file__).resolve().parents[1]
    frames_root = project_root / "data" / "frames"
    preds_root = project_root / "data" / "predictions"
    preds_root.mkdir(parents=True, exist_ok=True)

    # Auto-detect clips from frames directory
    if frames_root.exists():
        clips = sorted([d.name for d in frames_root.iterdir() if d.is_dir()])
        if not clips:
            print(f"[ERROR] No clips found in {frames_root}")
            return
        print(f"[INFO] Detected clips: {', '.join(clips)}")
    else:
        clips = ["clip1", "clip2", "clip3"]  # fallback

    wm = VLMWorldModel()

    out_path = preds_root / "world_state_claude.csv"

    fieldnames = [
        "clip",
        "frame_index",
        "frame_filename",
        "frame_path",
        "affordance",
        "yield_to",
        "lead_state",
        "explanation",
        "raw",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for clip in clips:
            clip_dir = frames_root / clip
            if not clip_dir.exists():
                print(f"Skipping {clip}: {clip_dir} does not exist")
                continue

            frames = list_frames(clip_dir)
            print(f"Processing {clip}: {len(frames)} frames")

            for idx, frame_path in enumerate(frames):
                print(f"[{clip}] {idx+1}/{len(frames)} -> {frame_path.name}")
                result = wm.infer_frame(frame_path)

                row = {
                    "clip": clip,
                    "frame_index": idx,
                    "frame_filename": frame_path.name,
                    "frame_path": str(frame_path),
                    "affordance": result.get("affordance", ""),
                    "yield_to": result.get("yield_to", ""),
                    "lead_state": result.get("lead_state", ""),
                    "explanation": result.get("explanation", ""),
                    "raw": result.get("raw", ""),
                }
                writer.writerow(row)

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
