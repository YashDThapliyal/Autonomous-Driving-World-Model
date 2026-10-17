#!/usr/bin/env python
"""
Simple script to count frames per clip and total.
"""

from pathlib import Path

def count_frames():
    frames_root = Path("data/frames")
    if not frames_root.exists():
        print(f"Error: {frames_root} does not exist")
        return
    
    total = 0
    clip_counts = {}
    
    for clip_dir in sorted(frames_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        
        clip_name = clip_dir.name
        jpgs = list(clip_dir.glob("frame_*.jpg"))
        pngs = list(clip_dir.glob("frame_*.png"))
        count = len(jpgs) + len(pngs)
        
        clip_counts[clip_name] = count
        total += count
        print(f"{clip_name}: {count} frames")
    
    print(f"\nTotal: {total} frames across {len(clip_counts)} clips")

if __name__ == "__main__":
    count_frames()
