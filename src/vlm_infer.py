# src/vlm_infer.py

import os
import base64
import time
from pathlib import Path
from typing import Dict, Optional

from anthropic import Anthropic


SYSTEM_PROMPT = """
You are a driving affordance world model that looks at single monocular frames from a forward-facing camera in a car.

Your job is to infer a compact, interpretable world state for the EGO vehicle (the viewer) using this fixed schema:

- affordance: one of {go, wait, stop}
- yield_to: one of {none, lead, ped}
- lead_state: one of {none, moving, stopped}

Semantics:

- "affordance":
  - go   → the corridor is open; it is reasonable for the ego vehicle to move forward.
  - wait → the ego vehicle is temporarily constrained or negotiating (e.g., waiting for a gap, slowing but not forced fully to a hard stop).
  - stop → the ego vehicle must be stopped due to a hard constraint (e.g., red light, pedestrian in crosswalk, fully blocked lane).

- "yield_to":
  - none → there is no clear negotiation target directly constraining the ego vehicle (no key lead car or pedestrian that must be yielded to).
  - lead → the ego vehicle is queueing or following another vehicle in-lane that constrains its motion.
  - ped  → a pedestrian has effective right-of-way over the ego car (e.g., crossing, about to cross, or clearly being yielded to).

- "lead_state":
  - none    → no in-lane leader that constrains the ego car’s motion.
  - moving  → the in-lane leader in front of the ego car is clearly moving forward.
  - stopped → the in-lane leader in front is clearly stopped (e.g., at a red light or in a queue).

Always reason from the point of view of the ego car’s lane and obligations:
- Use "ped" only if a pedestrian is clearly relevant to the ego’s motion.
- Use "lead" only for the primary car directly ahead in-lane, not random cars elsewhere.
- If the ego is clearly at rest due to a strong constraint (red light, pedestrian, blocked intersection), prefer affordance=stop instead of wait.
- Use "wait" for softer negotiation states: waiting for a safe gap, creeping, slowing but not forced to full stop yet.

You must output exactly one line with the three labels, and then one line with a brief explanation.
Do not invent new labels or extra fields.
""".strip()


USER_TEXT_PROMPT = """
Given this frame from a forward-facing driving video, infer the current world state for the ego vehicle.

Use this exact output format:

affordance=<go|wait|stop>; yield_to=<none|lead|ped>; lead_state=<none|moving|stopped>;
explanation: <one short sentence explaining why.>

Examples:

Example 1 (stopped at red light behind a car):
affordance=stop; yield_to=lead; lead_state=stopped;
explanation: The ego car is stopped behind another vehicle at a red light, waiting for the signal to change.

Example 2 (clear road on highway, following a moving leader):
affordance=go; yield_to=lead; lead_state=moving;
explanation: The ego car is moving along the highway following a vehicle ahead that is also moving.

Example 3 (stopped for pedestrian in a crosswalk):
affordance=stop; yield_to=ped; lead_state=none;
explanation: The ego car is stopped to let a pedestrian cross in front of it.

Now analyze the provided frame and respond in the same format.
""".strip()


def _load_image_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_world_state(text: str) -> Dict[str, str]:
    """
    Parse the model's response into a structured dict.

    Expected format:
    affordance=wait; yield_to=ped; lead_state=stopped;
    explanation: The ego vehicle is ...
    """
    out: Dict[str, str] = {
        "affordance": "",
        "yield_to": "",
        "lead_state": "",
        "explanation": "",
        "raw": text.strip(),
    }

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return out

    first = lines[0]
    if len(lines) > 1 and lines[1].lower().startswith("explanation:"):
        out["explanation"] = lines[1].split(":", 1)[1].strip()
    elif len(lines) > 1:
        out["explanation"] = lines[1]

    parts = [p.strip() for p in first.split(";") if p.strip()]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()

    # clamp to valid vocab
    valid_aff = {"go", "wait", "stop"}
    valid_yield = {"none", "lead", "ped"}
    valid_lead = {"none", "moving", "stopped"}

    if out["affordance"] not in valid_aff:
        out["affordance"] = ""
    if out["yield_to"] not in valid_yield:
        out["yield_to"] = ""
    if out["lead_state"] not in valid_lead:
        out["lead_state"] = ""

    return out


class VLMWorldModel:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-haiku-4-5-20251001",
    ):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in the environment.")
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name

    def infer_frame(self, image_path: Path) -> Dict[str, str]:
        print(f"[VLMWorldModel] Calling Claude for: {image_path}")  # debug log

        # Encode image
        image_b64 = _load_image_base64(image_path)

        # Pick media_type based on extension
        suffix = image_path.suffix.lower()
        if suffix in [".png"]:
            media_type = "image/png"
        else:
            # default to jpeg
            media_type = "image/jpeg"

        start = time.time()
        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=256,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_TEXT_PROMPT},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                    ],
                },
            ],
        )
        elapsed = time.time() - start
        print(f"[VLMWorldModel] Claude call completed in {elapsed:.2f}s for: {image_path.name}")

        # Add delay to avoid rate limiting (0.5 seconds between API calls)
        time.sleep(0.5)

        # Collect text blocks
        text_chunks = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_chunks.append(block.text)
        full_text = "\n".join(text_chunks).strip()

        parsed = parse_world_state(full_text)
        if not parsed.get("raw"):
            parsed["raw"] = full_text
        return parsed
