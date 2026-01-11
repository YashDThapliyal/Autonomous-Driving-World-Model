# src/generate_segment_gloss.py

import base64
import csv
import json
import os
from pathlib import Path
from typing import Tuple

from anthropic import Anthropic


MODEL_NAME = os.getenv("ANTHROPIC_SEGMENT_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are an autonomous driving semantics assistant.

You are given:
- a short temporal driving segment (ego car)
- a coarse "phase" label for that segment
- one representative frame image from that segment

Your job is to return TWO fields:

1) short_label: 3–7 words, planner-style, concise, no punctuation.
   - This will be shown in a HUD line as: "SEGMENT: <short_label>"
   - Examples:
     - "pedestrian crossing ahead"
     - "nighttime congestion queue"
     - "following moving leader"
     - "selects gap in left lane"

2) long_label: one short sentence (<= 25 words)
   - Planner-style explanation of what constrains or enables the ego vehicle.
   - Focus on what the ego is doing or waiting for (yielding, stopped, following, selecting gap, etc).

Return ONLY valid JSON with exactly:
{
  "short_label": "...",
  "long_label": "..."
}"""


def b64_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pick_rep_frame(start: int, end: int) -> int:
    """Pick a representative frame index (0-based) from a segment."""
    return (start + end) // 2


def build_user_prompt(clip: str, phase: str, seg_id: int, start: int, end: int) -> str:
    return f"""You are summarizing a short driving segment from clip '{clip}'.

Segment metadata:
- segment_id: {seg_id}
- phase: {phase}
- frame range (0-based indices): {start}–{end}

Please infer what is happening from the ego vehicle's perspective and follow the JSON spec exactly."""
    

def call_claude_for_segment(
    client: Anthropic,
    model_name: str,
    image_b64: str,
    clip: str,
    phase: str,
    seg_id: int,
    start: int,
    end: int,
) -> Tuple[str, str]:
    user_prompt = build_user_prompt(clip, phase, seg_id, start, end)

    msg = client.messages.create(
        model=model_name,
        max_tokens=200,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
    )

    # Expect first content block to be text JSON
    text = ""
    for block in msg.content:
        if block.type == "text":
            text = block.text
            break

    short_label = ""
    long_label = ""

    # Hardened fenced JSON cleanup for Claude outputs
    clean = text.strip()

    # Remove triple fences like ```json ... ```
    if clean.startswith("```"):
        # remove all backticks
        clean = clean.replace("```", "").replace("`", "").strip()
        # remove json lang tag
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()

    # Remove stray inline backticks
    clean = clean.replace("`", "").strip()

    short_label = ""
    long_label = ""

    # Try JSON decoding
    try:
        data = json.loads(clean)
        short_label = data.get("short_label", "").strip()
        long_label = data.get("long_label", "").strip()
    except Exception:
        # fallback: derive short label
        long_label = clean
        words = clean.split()
        short_label = " ".join(words[:5]).lower()

    # final defaults
    if not short_label:
        short_label = phase
    if not long_label:
        long_label = phase

    return short_label, long_label


def main():
    project_root = Path(__file__).resolve().parents[1]
    preds_root = project_root / "data" / "predictions"
    segments_path = preds_root / "world_state_segments.csv"
    frames_root = project_root / "data" / "frames"
    out_path = preds_root / "segment_gloss.csv"

    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    client = Anthropic()

    # Read segments
    with segments_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        segments = list(reader)

    # Prepare output
    fieldnames = [
        "clip",
        "segment_id",
        "start",
        "end",
        "affordance",
        "yield_to",
        "lead_state",
        "phase",
        "gloss_short",
        "gloss_long",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()

        for row in segments:
            clip = row["clip"]
            seg_id = int(row["segment_id"])
            start = int(row["start"])
            end = int(row["end"])
            affordance = row["affordance"]
            yield_to = row["yield_to"]
            lead_state = row["lead_state"]
            phase = row["phase"]

            # Compute representative frame index and path
            rep_idx = pick_rep_frame(start, end)  # 0-based index
            # Your frame files are frame_0001.jpg for index 0
            frame_filename = f"frame_{rep_idx + 1:04d}.jpg"
            frame_path = frames_root / clip / frame_filename

            if not frame_path.exists():
                print(f"[WARN] Representative frame not found: {frame_path}, skipping segment {clip}-{seg_id}")
                gloss_short = phase
                gloss_long = phase
            else:
                print(f"[INFO] Processing segment {clip}-{seg_id} using {frame_path}")
                image_b64 = b64_image(frame_path)
                gloss_short, gloss_long = call_claude_for_segment(
                    client=client,
                    model_name=MODEL_NAME,
                    image_b64=image_b64,
                    clip=clip,
                    phase=phase,
                    seg_id=seg_id,
                    start=start,
                    end=end,
                )

            writer.writerow(
                {
                    "clip": clip,
                    "segment_id": seg_id,
                    "start": start,
                    "end": end,
                    "affordance": affordance,
                    "yield_to": yield_to,
                    "lead_state": lead_state,
                    "phase": phase,
                    "gloss_short": gloss_short,
                    "gloss_long": gloss_long,
                }
            )

    print(f"\nSaved segment glosses to {out_path}")


if __name__ == "__main__":
    main()
