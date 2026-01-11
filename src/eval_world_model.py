# src/eval_world_model.py

import csv
from pathlib import Path
from typing import Dict, Tuple, List, Any


def parse_frame_id(frame_id: str) -> Tuple[str, int]:
    """
    Parse a frame_id like 'clip1/frame_0001.jpg' into:
      clip = 'clip1'
      frame_index = 0-based index (0001 -> 0, 0002 -> 1, etc.)
    """
    # Normalize separators just in case
    frame_id = frame_id.replace("\\", "/")

    parts = frame_id.split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected frame_id format: {frame_id}")

    clip = parts[0]  # e.g. 'clip1'
    filename = parts[-1]  # e.g. 'frame_0001.jpg'

    # Extract the numeric part from 'frame_0001'
    if not filename.startswith("frame_"):
        raise ValueError(f"Unexpected frame filename in frame_id: {frame_id}")

    base = filename.split(".")[0]  # 'frame_0001'
    num_str = base.replace("frame_", "")  # '0001'
    try:
        frame_num = int(num_str)
    except ValueError:
        raise ValueError(f"Could not parse frame number from frame_id: {frame_id}")

    # Our predictions use 0-based indices: frame_0001.jpg -> index 0
    frame_index = frame_num - 1
    return clip, frame_index


def load_labels(labels_dir: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    """
    Load human labels from all CSVs in data/labels.

    Expected columns in your files (based on error message):
      - frame_id  (e.g. 'clip1/frame_0001.jpg')
      - affordance
      - yield_to
      - lead_state
    """
    label_map: Dict[Tuple[str, int], Dict[str, str]] = {}

    csv_files = sorted(labels_dir.glob("labels_clip*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No label CSVs found in {labels_dir}")

    for path in csv_files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = row.get("frame_id")
                if frame_id is None:
                    raise ValueError(
                        f"Missing frame_id in {path}: {row} "
                        "(expected a 'frame_id' column like 'clip1/frame_0001.jpg')"
                    )

                clip, frame_index = parse_frame_id(frame_id)
                key = (clip, frame_index)

                label_map[key] = {
                    "affordance": row.get("affordance", "").strip(),
                    "yield_to": row.get("yield_to", "").strip(),
                    "lead_state": row.get("lead_state", "").strip(),
                }

    return label_map


def load_predictions(preds_path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    """
    Load smoothed predictions from world_state_claude_smoothed.csv.

    Expected columns:
      - clip
      - frame_index
      - affordance_smoothed
      - yield_to_smoothed
      - lead_state_smoothed
    """
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    pred_map: Dict[Tuple[str, int], Dict[str, str]] = {}

    with preds_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            frame_index = int(row["frame_index"])
            key = (clip, frame_index)
            pred_map[key] = {
                "affordance": row.get("affordance_smoothed", row.get("affordance", "")).strip(),
                "yield_to": row.get("yield_to_smoothed", row.get("yield_to", "")).strip(),
                "lead_state": row.get("lead_state_smoothed", row.get("lead_state", "")).strip(),
            }

    return pred_map


def compute_metrics(
    labels: Dict[Tuple[str, int], Dict[str, str]],
    preds: Dict[Tuple[str, int], Dict[str, str]],
) -> Dict[str, Any]:
    """
    Compute per-axis accuracy over frames where both label and pred exist.
    """
    total = 0
    correct_aff = 0
    correct_yield = 0
    correct_lead = 0

    mismatches: List[Dict[str, Any]] = []

    for key, l in labels.items():
        if key not in preds:
            continue
        p = preds[key]
        total += 1

        if l["affordance"] == p["affordance"]:
            correct_aff += 1
        else:
            mismatches.append(
                {
                    "clip": key[0],
                    "frame_index": key[1],
                    "axis": "affordance",
                    "label": l["affordance"],
                    "pred": p["affordance"],
                }
            )

        if l["yield_to"] == p["yield_to"]:
            correct_yield += 1
        else:
            mismatches.append(
                {
                    "clip": key[0],
                    "frame_index": key[1],
                    "axis": "yield_to",
                    "label": l["yield_to"],
                    "pred": p["yield_to"],
                }
            )

        if l["lead_state"] == p["lead_state"]:
            correct_lead += 1
        else:
            mismatches.append(
                {
                    "clip": key[0],
                    "frame_index": key[1],
                    "axis": "lead_state",
                    "label": l["lead_state"],
                    "pred": p["lead_state"],
                }
            )

    metrics = {
        "total_frames_with_labels": total,
        "affordance_acc": correct_aff / total if total else 0.0,
        "yield_to_acc": correct_yield / total if total else 0.0,
        "lead_state_acc": correct_lead / total if total else 0.0,
        "mismatches": mismatches,
    }
    return metrics


def print_sample_table(
    clip: str,
    labels: Dict[Tuple[str, int], Dict[str, str]],
    preds: Dict[Tuple[str, int], Dict[str, str]],
    max_rows: int = 20,
):
    """
    Print a small side-by-side table for a given clip.
    """
    print(f"\n=== Sample comparison for {clip} ===")
    rows: List[Tuple[int, Dict[str, str], Dict[str, str]]] = []

    for (c, fi), l in labels.items():
        if c != clip:
            continue
        if (c, fi) not in preds:
            continue
        rows.append((fi, l, preds[(c, fi)]))

    rows.sort(key=lambda x: x[0])

    print(
        f"{'frame':>5} | {'aff_label':>10} | {'aff_pred':>10} | "
        f"{'y_label':>8} | {'y_pred':>8} | {'lead_label':>10} | {'lead_pred':>10}"
    )
    print("-" * 80)

    for fi, l, p in rows[:max_rows]:
        print(
            f"{fi:5d} | "
            f"{l['affordance']:>10} | {p['affordance']:>10} | "
            f"{l['yield_to']:>8} | {p['yield_to']:>8} | "
            f"{l['lead_state']:>10} | {p['lead_state']:>10}"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]
    labels_dir = project_root / "data" / "labels"
    preds_path = project_root / "data" / "predictions" / "world_state_claude_smoothed.csv"

    print(f"Loading labels from: {labels_dir}")
    labels = load_labels(labels_dir)

    print(f"Loading predictions from: {preds_path}")
    preds = load_predictions(preds_path)

    metrics = compute_metrics(labels, preds)

    print("\n=== World Model Accuracy (per-axis) ===")
    print(f"Frames with labels: {metrics['total_frames_with_labels']}")
    print(f"Affordance accuracy: {metrics['affordance_acc']:.3f}")
    print(f"Yield_to accuracy : {metrics['yield_to_acc']:.3f}")
    print(f"Lead_state accuracy: {metrics['lead_state_acc']:.3f}")

    # Print small example table for clip1 if it exists
    clips_present = sorted({c for (c, _) in labels.keys()})
    if "clip1" in clips_present:
        print_sample_table("clip1", labels, preds, max_rows=20)
    else:
        if clips_present:
            print_sample_table(clips_present[0], labels, preds, max_rows=20)

    # (Optional) show a few mismatches
    mismatches = metrics["mismatches"]
    if mismatches:
        print("\n=== Example mismatches (up to 10) ===")
        for m in mismatches[:10]:
            print(
                f"clip={m['clip']}, frame={m['frame_index']}, axis={m['axis']}, "
                f"label={m['label']}, pred={m['pred']}"
            )


if __name__ == "__main__":
    main()
