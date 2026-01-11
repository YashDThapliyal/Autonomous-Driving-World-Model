# src/overlay_world_state.py

import argparse
from pathlib import Path
import csv
from typing import Dict, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from run_yolo import run_yolo, filter_peds, filter_cars

# ============================================================================
# YOLO Target Selection Heuristic Weights
# ============================================================================
# Pedestrian selection weights (lower score = better candidate)
PED_W1_X_CENTER = 0.4  # Weight for horizontal distance from center
PED_W2_VERTICAL = 0.3  # Weight for vertical position (prefer lower in frame)
PED_W3_BBOX_HEIGHT = 0.3  # Weight for bbox height (prefer larger/taller)

# Lead vehicle selection weights (lower score = better candidate)
LEAD_W1_Y2 = 0.4  # Weight for y2 position (prefer smaller = closer forward)
LEAD_W2_HORIZONTAL = 0.3  # Weight for horizontal alignment (prefer center)
LEAD_W3_HEIGHT = 0.3  # Weight for inverse height (prefer larger = closer)

# ROI filters (as fractions of frame dimensions)
PED_ROI_TOP_EXCLUDE = 0.20  # Exclude top 20% of frame (sky/buildings)
PED_ROI_BOTTOM_EXCLUDE = 0.10  # Exclude bottom 10% (too close/occluded)
PED_ROI_CENTER_WIDTH = 0.33  # Prefer center 1/3 width (front-of-lane)

LEAD_ROI_CENTER_WIDTH = 0.50  # Include center 50% width for lead cars


def load_smoothed_preds(csv_path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    pred_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            fi = int(row["frame_index"])
            key = (clip, fi)
            pred_map[key] = {
                "affordance": row.get("affordance_smoothed", row.get("affordance", "")),
                "yield_to": row.get("yield_to_smoothed", row.get("yield_to", "")),
                "lead_state": row.get("lead_state_smoothed", row.get("lead_state", "")),
            }
    return pred_map


def load_segments(segments_path: Path):
    by_clip = {}
    with segments_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            by_clip.setdefault(clip, []).append({
                "segment_id": int(row["segment_id"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "phase": row["phase"],
            })
    for clip in by_clip:
        by_clip[clip].sort(key=lambda s: s["start"])
    return by_clip


def load_gloss(gloss_path: Path):
    gmap = {}
    with gloss_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["clip"], int(row["segment_id"]))
            gmap[key] = {
                "gloss_short": row.get("gloss_short", "").strip(),
                "gloss_long": row.get("gloss_long", "").strip(),
            }
    return gmap


def load_planner(planner_path: Path):
    """
    Returns (clip, segment_id) -> planner dict:
    {
      behavior: FOLLOW / STOP / WAIT / YIELD (etc)
      intent:   FOLLOW(lead) until lead_moves
      target:   lead / ped / "" (for YOLO highlighting)
    }
    """
    pmap = {}
    with planner_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            seg_id = int(row["segment_id"])
            behavior = row.get("planner_cmd", "").strip().upper()

            target = row.get("target", "").strip()
            until = row.get("until", "").strip()

            if target and until:
                intent = f"{behavior.title()}({target}) until {until}"
            elif target:
                intent = f"{behavior.title()}({target})"
            else:
                intent = behavior.title()

            pmap[(clip, seg_id)] = {
                "behavior": behavior,
                "intent": intent,
                "target": target,
            }
    return pmap


def find_segment(clip: str, frame_index: int, segments) -> Tuple[int, str]:
    segs = segments.get(clip, [])
    for seg in segs:
        if seg["start"] <= frame_index <= seg["end"]:
            return seg["segment_id"], seg["phase"]
    return -1, "UNKNOWN"


def collect_frame_paths(frames_dir: Path):
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.jpg or .png found in {frames_dir}")
    return paths


def draw_hud(img, clip_name, frame_index, state):
    if img.mode != "RGB":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    W, H = img.size

    base_font_size = max(18, min(40, H // 28))
    try:
        font = ImageFont.truetype("Menlo.ttf", base_font_size)
    except Exception:
        font = ImageFont.load_default()

    # HUD lines (clean cognitive order)
    lines = [
        f"CLIP: {clip_name}  FRAME: {frame_index}",
        f"PHASE: {state.get('phase','N/A')}",
    ]

    gs = state.get("gloss_short", "")
    if gs:
        lines.append(f"SCENE: {gs}")

    lines.append(f"AFFORDANCE: {state.get('affordance','N/A')}")
    lines.append(f"YIELD TO:   {state.get('yield_to','N/A')}")
    lines.append(f"LEAD STATE: {state.get('lead_state','N/A')}")

    behavior = state.get("behavior","")
    if behavior:
        lines.append(f"PLAN: {behavior}")

    intent = state.get("intent","")
    if intent:
        lines.append(f"INTENT: {intent}")

    def text_size(line):
        bbox = draw.textbbox((0, 0), line, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]

    text_w = 0
    text_h = 0
    for line in lines:
        w, h = text_size(line)
        text_w = max(text_w, w)
        text_h += h + 6

    box_w = text_w + 40
    box_h = text_h + 40
    box_coords = (10, 10, 10 + box_w, 10 + box_h)

    draw.rectangle(box_coords, fill=(0, 0, 0))
    y = box_coords[1] + 20
    x = box_coords[0] + 20
    for line in lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        _, h = text_size(line)
        y += h + 6

    return img


def draw_yolo_target(img, target_box, object_label):
    """
    Draw a green box and label around the target object.
    
    Args:
        img: PIL Image (RGB)
        target_box: dict with 'xyxy' key containing (x1, y1, x2, y2)
        object_label: string label for the object (e.g., "PEDESTRIAN", "CAR")
    
    Returns:
        PIL Image with box drawn
    """
    # Convert PIL to numpy array for cv2
    img_np = np.array(img)
    
    x1, y1, x2, y2 = target_box['xyxy']
    
    # Draw green rectangle (BGR format for cv2)
    color = (0, 255, 0)  # Green in BGR
    thickness = 2
    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label above box
    label = object_label.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    label_thickness = 2
    
    # Get label size for positioning
    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
    
    # Position label above box, centered horizontally
    label_x = x1
    label_y = max(y1 - 5, label_height + 5)  # Position above box with small margin
    
    # Draw text with background for readability
    cv2.rectangle(img_np, 
                  (label_x - 2, label_y - label_height - 2),
                  (label_x + label_width + 2, label_y + baseline + 2),
                  (0, 0, 0), -1)  # Black background
    cv2.putText(img_np, label, (label_x, label_y), font, font_scale, color, label_thickness)
    
    # Convert back to PIL Image
    return Image.fromarray(img_np)


def select_pedestrian_target(peds: list, frame_width: int, frame_height: int) -> Dict:
    """
    Select best pedestrian target using improved heuristics.
    
    Args:
        peds: List of pedestrian box dicts from filter_peds()
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Best pedestrian box dict, or None if no candidates
    """
    if not peds:
        return None
    
    # Filter by vertical ROI (exclude top and bottom regions)
    top_exclude_y = int(frame_height * PED_ROI_TOP_EXCLUDE)
    bottom_exclude_y = int(frame_height * (1 - PED_ROI_BOTTOM_EXCLUDE))
    
    filtered_peds = []
    for ped in peds:
        x1, y1, x2, y2 = ped['xyxy']
        y_center = (y1 + y2) / 2
        # Keep if in vertical ROI
        if top_exclude_y <= y_center <= bottom_exclude_y:
            filtered_peds.append(ped)
    
    # If filtering removed all candidates, fall back to all peds
    if not filtered_peds:
        filtered_peds = peds
    
    # Filter by horizontal ROI (prefer center region)
    center_left = frame_width * (0.5 - PED_ROI_CENTER_WIDTH / 2)
    center_right = frame_width * (0.5 + PED_ROI_CENTER_WIDTH / 2)
    
    center_peds = []
    for ped in filtered_peds:
        x1, y1, x2, y2 = ped['xyxy']
        x_center = (x1 + x2) / 2
        if center_left <= x_center <= center_right:
            center_peds.append(ped)
    
    # Use center region if available, otherwise use filtered
    candidates = center_peds if center_peds else filtered_peds
    
    # Score each candidate (lower is better)
    frame_center_x = frame_width / 2
    scores = []
    
    for ped in candidates:
        x1, y1, x2, y2 = ped['xyxy']
        x_center = (x1 + x2) / 2
        bbox_height = y2 - y1
        
        # Normalize features to [0, 1] range
        x_dist_norm = abs(x_center - frame_center_x) / (frame_width / 2)  # 0 = center, 1 = edge
        y2_raw = y2 / frame_height  # 0 = top, 1 = bottom
        vertical_proximity = 1.0 - y2_raw  # Invert: 1 = top (far), 0 = bottom (close) - prefer close
        height_norm = 1.0 - (bbox_height / frame_height)  # 0 = very tall, 1 = very short (prefer tall)
        
        # Combined score (lower is better)
        score = (PED_W1_X_CENTER * x_dist_norm + 
                PED_W2_VERTICAL * vertical_proximity + 
                PED_W3_BBOX_HEIGHT * height_norm)
        scores.append((score, ped))
    
    # Return ped with lowest score
    scores.sort(key=lambda x: x[0])
    return scores[0][1] if scores else None


def select_lead_car_target(cars: list, frame_width: int, frame_height: int) -> Dict:
    """
    Select best lead car target using improved heuristics.
    
    Args:
        cars: List of car box dicts from filter_cars()
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Best car box dict, or None if no candidates
    """
    if not cars:
        return None
    
    # Filter by horizontal ROI (exclude cars too far to sides)
    center_left = frame_width * (0.5 - LEAD_ROI_CENTER_WIDTH / 2)
    center_right = frame_width * (0.5 + LEAD_ROI_CENTER_WIDTH / 2)
    
    filtered_cars = []
    for car in cars:
        x1, y1, x2, y2 = car['xyxy']
        x_center = (x1 + x2) / 2
        if center_left <= x_center <= center_right:
            filtered_cars.append(car)
    
    # If filtering removed all candidates, fall back to all cars
    candidates = filtered_cars if filtered_cars else cars
    
    # Score each candidate (lower is better)
    frame_center_x = frame_width / 2
    scores = []
    
    # Find min/max for normalization
    y2_values = [car['xyxy'][3] for car in candidates]
    heights = [car['xyxy'][3] - car['xyxy'][1] for car in candidates]
    min_y2, max_y2 = min(y2_values), max(y2_values)
    min_h, max_h = min(heights), max(heights)
    
    for car in candidates:
        x1, y1, x2, y2 = car['xyxy']
        x_center = (x1 + x2) / 2
        bbox_height = y2 - y1
        
        # Normalize features to [0, 1] range
        if max_y2 > min_y2:
            y2_norm = (y2 - min_y2) / (max_y2 - min_y2)  # 0 = closest, 1 = farthest
        else:
            y2_norm = 0.5
        
        x_dist_norm = abs(x_center - frame_center_x) / (frame_width / 2)  # 0 = center, 1 = edge
        
        if max_h > min_h and bbox_height > 0:
            inv_height_norm = 1.0 - ((bbox_height - min_h) / (max_h - min_h))  # 0 = largest, 1 = smallest (prefer large)
        else:
            inv_height_norm = 0.5
        
        # Combined score (lower is better)
        score = (LEAD_W1_Y2 * y2_norm + 
                LEAD_W2_HORIZONTAL * x_dist_norm + 
                LEAD_W3_HEIGHT * inv_height_norm)
        scores.append((score, car))
    
    # Return car with lowest score
    scores.sort(key=lambda x: x[0])
    return scores[0][1] if scores else None


def make_overlay_video_for_clip(clip_name, frames_root, preds, segments, gloss, planner, out_dir, fps=2):
    frames_dir = frames_root / clip_name
    frame_paths = collect_frame_paths(frames_dir)

    frames = []
    for idx, frame_path in enumerate(frame_paths):
        frame_index = idx
        key = (clip_name, frame_index)

        base = preds.get(key, {"affordance":"N/A","yield_to":"N/A","lead_state":"N/A"})

        seg_id, phase = find_segment(clip_name, frame_index, segments)
        gloss_short = ""
        behavior = ""
        intent = ""
        target = ""

        if seg_id >= 0:
            g = gloss.get((clip_name, seg_id))
            if g:
                gloss_short = g.get("gloss_short", "")
            p = planner.get((clip_name, seg_id))
            if p:
                behavior = p.get("behavior","")
                intent = p.get("intent","")
                target = p.get("target","")

        state = {
            **base,
            "phase": phase,
            "gloss_short": gloss_short,
            "behavior": behavior,
            "intent": intent,
        }

        img = Image.open(frame_path).convert("RGB")
        
        # Run YOLO and highlight target if needed
        if behavior and target:
            # Convert PIL to numpy for YOLO
            img_np = np.array(img)
            
            # Run YOLO inference
            boxes = run_yolo(img_np)
            
            frame_width = img.size[0]
            frame_height = img.size[1]
            
            # Determine which objects to highlight based on target and behavior
            target_box = None
            object_label = None
            
            if target == "ped" and behavior in ["WAIT", "STOP"]:
                # Highlight pedestrian using improved heuristics
                peds = filter_peds(boxes)
                target_box = select_pedestrian_target(peds, frame_width, frame_height)
                if target_box:
                    object_label = "PEDESTRIAN"
            
            elif target == "lead" and behavior in ["FOLLOW", "STOP"]:
                # Highlight car using improved heuristics
                cars = filter_cars(boxes)
                target_box = select_lead_car_target(cars, frame_width, frame_height)
                if target_box:
                    object_label = "CAR"
            
            # Draw target box if found
            if target_box and object_label:
                img = draw_yolo_target(img, target_box, object_label)
        
        img = draw_hud(img, clip_name, frame_index, state)
        frames.append(img)

    frames_np = [np.array(im) for im in frames]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_name}_overlay.mp4"

    clip = ImageSequenceClip(frames_np, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264")
    print(f"Saved overlay video to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate overlay videos with world state HUD for driving clips"
    )
    parser.add_argument(
        "--clip",
        type=str,
        default="clip1",
        help="Clip name to process (e.g., clip1, clip2, clip3). Default: clip1",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for output video. Default: 2.0",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    preds_path = project_root / "data" / "predictions" / "world_state_claude_smoothed.csv"
    segments_path = project_root / "data" / "predictions" / "world_state_segments.csv"
    gloss_path = project_root / "data" / "predictions" / "segment_gloss.csv"
    planner_path = project_root / "data" / "predictions" / "planner_commands.csv"

    frames_root = project_root / "data" / "frames"
    out_dir = project_root / "results"

    preds = load_smoothed_preds(preds_path)
    segments = load_segments(segments_path)
    gloss = load_gloss(gloss_path)
    planner = load_planner(planner_path)

    make_overlay_video_for_clip(args.clip, frames_root, preds, segments, gloss, planner, out_dir, fps=args.fps)


if __name__ == "__main__":
    main()
