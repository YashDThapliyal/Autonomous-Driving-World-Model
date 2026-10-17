#!/usr/bin/env python
"""
Compute metrics for the Driving World Model project.

Enhanced analytics including:
- Clip-level timing & throughput
- Segment semantics & transitions
- World-state dynamics (transition matrices, entropy, dwell times)
- Planner performance
- Pipeline compression metrics
- YOLO visual grounding (enhanced)
"""

import argparse
from pathlib import Path
import sys
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# ----------------------------
# Setup paths
# ----------------------------

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions"
FRAMES_ROOT = ROOT / "data" / "frames"
VIDEOS_ROOT = ROOT / "data" / "raw_videos"

RAW_PATH = PRED_DIR / "world_state_claude.csv"
SMOOTH_PATH = PRED_DIR / "world_state_claude_smoothed.csv"
SEGMENTS_PATH = PRED_DIR / "world_state_segments.csv"
PLANNER_PATH = PRED_DIR / "planner_commands.csv"

# ----------------------------
# Optional YOLO import
# ----------------------------

YOLO_AVAILABLE = False
try:
    sys.path.append(str(ROOT / "src"))
    from run_yolo import run_yolo, filter_peds, filter_cars  # type: ignore
    import cv2
    YOLO_AVAILABLE = True
except Exception as e:
    print("[INFO] YOLO metrics disabled (could not import run_yolo / cv2).")
    print(f"       Reason: {e}\n")

# ----------------------------
# Optional cv2 for video metadata (if not already imported)
# ----------------------------

CV2_AVAILABLE = YOLO_AVAILABLE
if not CV2_AVAILABLE:
    try:
        import cv2
        CV2_AVAILABLE = True
    except Exception:
        CV2_AVAILABLE = False

# ----------------------------
# Utility functions
# ----------------------------

def count_flips(series):
    """Counts frame-to-frame label changes."""
    flips = 0
    prev = None
    for v in series:
        if prev is not None and v != prev:
            flips += 1
        prev = v
    return flips


def run_lengths(series):
    """Returns list of run lengths for consecutive identical values."""
    runs = []
    if not len(series):
        return runs
    current = series[0]
    length = 1
    for v in series[1:]:
        if v == current:
            length += 1
        else:
            runs.append(length)
            current = v
            length = 1
    runs.append(length)
    return runs


def entropy(values):
    """Compute Shannon entropy of a sequence."""
    counts = Counter(values)
    total = len(values)
    if total == 0:
        return 0.0
    probs = [count / total for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def transition_matrix(series):
    """Compute transition matrix from a sequence."""
    transitions = defaultdict(int)
    total_transitions = 0
    for i in range(len(series) - 1):
        from_state = series[i]
        to_state = series[i + 1]
        transitions[(from_state, to_state)] += 1
        total_transitions += 1
    
    # Get unique states
    states = sorted(set(series))
    matrix = {}
    for from_state in states:
        for to_state in states:
            count = transitions.get((from_state, to_state), 0)
            prob = count / total_transitions if total_transitions > 0 else 0.0
            matrix[(from_state, to_state)] = {
                'count': count,
                'prob': prob
            }
    return matrix, states


def collect_frames_per_clip(frames_root: Path):
    """Return dict: clip_name -> #frames."""
    frames_per_clip = {}
    for clip_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        jpgs = sorted(clip_dir.glob("frame_*.jpg"))
        pngs = sorted(clip_dir.glob("frame_*.png"))
        files = jpgs if jpgs else pngs
        frames_per_clip[clip_dir.name] = len(files)
    return frames_per_clip


def get_video_metadata(video_path: Path):
    """Extract FPS and duration from video file."""
    if not CV2_AVAILABLE or not video_path.exists():
        return None, None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0 or frame_count <= 0:
        return None, None
    
    duration = frame_count / fps if fps > 0 else None
    return fps, duration


def compute_clip_timing(frames_root: Path, videos_root: Path):
    """Compute timing and throughput metrics per clip."""
    frames_per_clip = collect_frames_per_clip(frames_root)
    timing_stats = {}
    
    for clip_name, n_frames in frames_per_clip.items():
        video_path = videos_root / f"{clip_name}.mp4"
        fps, duration = get_video_metadata(video_path)
        
        stats = {
            'n_frames': n_frames,
            'video_fps': fps,
            'video_duration_sec': duration,
        }
        
        if fps and duration:
            # Estimated time span covered by frames
            # Assuming frames were extracted at ~2fps (default)
            estimated_time_span = n_frames / 2.0  # seconds
            stats['estimated_time_span_sec'] = estimated_time_span
            stats['frames_per_sec'] = n_frames / estimated_time_span if estimated_time_span > 0 else 0.0
        else:
            stats['estimated_time_span_sec'] = None
            stats['frames_per_sec'] = None
        
        timing_stats[clip_name] = stats
    
    return timing_stats


def compute_temporal_metrics(raw_df: pd.DataFrame,
                             smooth_df: pd.DataFrame,
                             field_raw: str,
                             field_smooth: str):
    """
    Compute flip counts and run-length stats per clip and globally
    for a single field (e.g., affordance, yield_to, lead_state).
    """
    per_clip = {}
    total_raw_flips = 0
    total_smooth_flips = 0
    all_raw_runs = []
    all_smooth_runs = []

    # Ensure we only use shared (clip, frame_index) pairs
    merged = raw_df.merge(
        smooth_df,
        on=["clip", "frame_index"],
        suffixes=("_raw", "_smooth")
    )

    for clip, group in merged.groupby("clip"):
        group = group.sort_values("frame_index")
        raw_series = group[f"{field_raw}_raw"].tolist()
        # Smoothed column name is already complete (e.g., "affordance_smoothed")
        # No suffix added since column names differ between raw and smooth
        smooth_series = group[field_smooth].tolist()

        raw_flips = count_flips(raw_series)
        smooth_flips = count_flips(smooth_series)

        raw_runs = run_lengths(raw_series)
        smooth_runs = run_lengths(smooth_series)

        total_raw_flips += raw_flips
        total_smooth_flips += smooth_flips
        all_raw_runs.extend(raw_runs)
        all_smooth_runs.extend(smooth_runs)

        jitter_reduction = 0.0
        if raw_flips > 0:
            jitter_reduction = 100.0 * (raw_flips - smooth_flips) / raw_flips

        per_clip[clip] = {
            "raw_flips": raw_flips,
            "smooth_flips": smooth_flips,
            "jitter_reduction_pct": jitter_reduction,
            "mean_raw_run": sum(raw_runs) / len(raw_runs) if raw_runs else 0.0,
            "mean_smooth_run": sum(smooth_runs) / len(smooth_runs) if smooth_runs else 0.0,
        }

    global_jitter_reduction = 0.0
    if total_raw_flips > 0:
        global_jitter_reduction = 100.0 * (total_raw_flips - total_smooth_flips) / total_raw_flips

    global_mean_raw_run = sum(all_raw_runs) / len(all_raw_runs) if all_raw_runs else 0.0
    global_mean_smooth_run = sum(all_smooth_runs) / len(all_smooth_runs) if all_smooth_runs else 0.0

    global_stats = {
        "raw_flips": total_raw_flips,
        "smooth_flips": total_smooth_flips,
        "jitter_reduction_pct": global_jitter_reduction,
        "mean_raw_run": global_mean_raw_run,
        "mean_smooth_run": global_mean_smooth_run,
    }

    return per_clip, global_stats


def compute_world_state_dynamics(smooth_df: pd.DataFrame, field: str):
    """Compute transition matrices, entropy, and dwell times for a world state field."""
    field_col = f"{field}_smoothed" if f"{field}_smoothed" in smooth_df.columns else field
    
    if field_col not in smooth_df.columns:
        return {}, {}
    
    per_clip_stats = {}
    all_values = []
    
    for clip, group in smooth_df.groupby("clip"):
        group = group.sort_values("frame_index")
        values = group[field_col].tolist()
        all_values.extend(values)
        
        # Transition matrix
        trans_matrix, states = transition_matrix(values)
        
        # Entropy
        field_entropy = entropy(values)
        
        # Dwell times (run lengths)
        dwell_times = run_lengths(values)
        avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0.0
        min_dwell = min(dwell_times) if dwell_times else 0
        max_dwell = max(dwell_times) if dwell_times else 0
        
        per_clip_stats[clip] = {
            'entropy': field_entropy,
            'avg_dwell_time': avg_dwell,
            'min_dwell_time': min_dwell,
            'max_dwell_time': max_dwell,
            'transition_matrix': trans_matrix,
            'states': states,
        }
    
    # Global stats
    global_entropy = entropy(all_values)
    global_dwell_times = run_lengths(all_values)
    global_avg_dwell = sum(global_dwell_times) / len(global_dwell_times) if global_dwell_times else 0.0
    global_trans_matrix, global_states = transition_matrix(all_values)
    
    global_stats = {
        'entropy': global_entropy,
        'avg_dwell_time': global_avg_dwell,
        'min_dwell_time': min(global_dwell_times) if global_dwell_times else 0,
        'max_dwell_time': max(global_dwell_times) if global_dwell_times else 0,
        'transition_matrix': global_trans_matrix,
        'states': global_states,
        'unique_states': len(set(all_values)),
    }
    
    return per_clip_stats, global_stats


def compute_segment_semantics(segments_df: pd.DataFrame):
    """Compute segment type frequencies and transitions."""
    per_clip_stats = {}
    all_phases = []
    all_transitions = []
    
    for clip, group in segments_df.groupby("clip"):
        group = group.sort_values("segment_id")
        phases = group["phase"].tolist()
        all_phases.extend(phases)
        
        # Phase frequencies
        phase_freq = Counter(phases)
        
        # Transitions between segments
        transitions = []
        for i in range(len(phases) - 1):
            transitions.append((phases[i], phases[i + 1]))
        transition_freq = Counter(transitions)
        all_transitions.extend(transitions)
        
        # Segment lengths
        lengths = (group["end"] - group["start"] + 1).tolist()
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        
        per_clip_stats[clip] = {
            'phase_freq': phase_freq,
            'transition_freq': transition_freq,
            'avg_segment_length': avg_length,
            'n_segments': len(phases),
        }
    
    # Global stats
    global_phase_freq = Counter(all_phases)
    global_transition_freq = Counter(all_transitions)
    
    global_stats = {
        'phase_freq': global_phase_freq,
        'transition_freq': global_transition_freq,
        'unique_phases': len(set(all_phases)),
    }
    
    return per_clip_stats, global_stats


def compute_planner_performance(planner_df: pd.DataFrame):
    """Compute planner command transitions, dwell times, and diversity."""
    planner_df = planner_df.copy()
    planner_df["planner_cmd_norm"] = planner_df["planner_cmd"].astype(str).str.upper().str.strip()
    
    per_clip_stats = {}
    all_commands = []
    
    for clip, group in planner_df.groupby("clip"):
        group = group.sort_values("segment_id")
        commands = group["planner_cmd_norm"].tolist()
        all_commands.extend(commands)
        
        # Command frequencies
        cmd_freq = Counter(commands)
        
        # Transitions
        transitions = []
        for i in range(len(commands) - 1):
            transitions.append((commands[i], commands[i + 1]))
        transition_freq = Counter(transitions)
        
        # Dwell times (segment lengths)
        lengths = (group["end_frame"] - group["start_frame"] + 1).tolist()
        avg_dwell = sum(lengths) / len(lengths) if lengths else 0.0
        
        per_clip_stats[clip] = {
            'cmd_freq': cmd_freq,
            'transition_freq': transition_freq,
            'avg_dwell_frames': avg_dwell,
            'unique_commands': len(set(commands)),
        }
    
    # Global stats
    global_cmd_freq = Counter(all_commands)
    global_transitions = []
    for clip, group in planner_df.groupby("clip"):
        group = group.sort_values("segment_id")
        commands = group["planner_cmd_norm"].tolist()
        for i in range(len(commands) - 1):
            global_transitions.append((commands[i], commands[i + 1]))
    global_transition_freq = Counter(global_transitions)
    
    global_stats = {
        'cmd_freq': global_cmd_freq,
        'transition_freq': global_transition_freq,
        'unique_commands': len(set(all_commands)),
    }
    
    return per_clip_stats, global_stats


def compute_yolo_grounding(frames_root: Path,
                           segments_df: pd.DataFrame,
                           planner_df: pd.DataFrame):
    """
    Enhanced YOLO grounding with confusion categories:
    - correct target (correct type detected)
    - wrong target type (wrong type detected)
    - none (no relevant detection)
    - ambiguous (multiple candidates)
    """
    if not YOLO_AVAILABLE:
        return {}, {}
    
    # Build lookup: (clip, segment_id) -> (start, end)
    seg_map = {}
    for _, row in segments_df.iterrows():
        seg_map[(row["clip"], int(row["segment_id"]))] = (
            int(row["start"]),
            int(row["end"]),
        )
    
    per_clip_counts = defaultdict(lambda: {
        "applicable": 0,
        "correct": 0,
        "wrong_type": 0,
        "none": 0,
        "ambiguous": 0
    })
    
    total_counts = {
        "applicable": 0,
        "correct": 0,
        "wrong_type": 0,
        "none": 0,
        "ambiguous": 0
    }
    
    for _, row in planner_df.iterrows():
        clip = row["clip"]
        seg_id = int(row["segment_id"])
        target = str(row.get("target", "")).strip().lower()
        
        if target not in {"ped", "lead"}:
            continue
        if (clip, seg_id) not in seg_map:
            continue
        
        start, end = seg_map[(clip, seg_id)]
        mid_index = (start + end) // 2
        frame_fname = f"frame_{mid_index + 1:04d}.jpg"
        frame_path = frames_root / clip / frame_fname
        if not frame_path.exists():
            frame_fname = f"frame_{mid_index + 1:04d}.png"
            frame_path = frames_root / clip / frame_fname
            if not frame_path.exists():
                continue
        
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        boxes = run_yolo(img_rgb)
        peds = filter_peds(boxes)
        cars = filter_cars(boxes)
        
        per_clip_counts[clip]["applicable"] += 1
        total_counts["applicable"] += 1
        
        if target == "ped":
            n_peds = len(peds)
            n_cars = len(cars)
            if n_peds > 1:
                per_clip_counts[clip]["ambiguous"] += 1
                total_counts["ambiguous"] += 1
            elif n_peds == 1:
                per_clip_counts[clip]["correct"] += 1
                total_counts["correct"] += 1
            elif n_cars > 0:
                per_clip_counts[clip]["wrong_type"] += 1
                total_counts["wrong_type"] += 1
            else:
                per_clip_counts[clip]["none"] += 1
                total_counts["none"] += 1
        elif target == "lead":
            n_cars = len(cars)
            n_peds = len(peds)
            if n_cars > 1:
                per_clip_counts[clip]["ambiguous"] += 1
                total_counts["ambiguous"] += 1
            elif n_cars == 1:
                per_clip_counts[clip]["correct"] += 1
                total_counts["correct"] += 1
            elif n_peds > 0:
                per_clip_counts[clip]["wrong_type"] += 1
                total_counts["wrong_type"] += 1
            else:
                per_clip_counts[clip]["none"] += 1
                total_counts["none"] += 1
    
    per_clip_stats = {}
    for clip, stats in per_clip_counts.items():
        applicable = stats["applicable"]
        per_clip_stats[clip] = {
            "applicable": applicable,
            "correct": stats["correct"],
            "wrong_type": stats["wrong_type"],
            "none": stats["none"],
            "ambiguous": stats["ambiguous"],
            "correct_pct": 100.0 * stats["correct"] / applicable if applicable > 0 else 0.0,
        }
    
    global_stats = {}
    if total_counts["applicable"] > 0:
        global_stats = {
            "applicable": total_counts["applicable"],
            "correct": total_counts["correct"],
            "wrong_type": total_counts["wrong_type"],
            "none": total_counts["none"],
            "ambiguous": total_counts["ambiguous"],
            "correct_pct": 100.0 * total_counts["correct"] / total_counts["applicable"],
        }
    
    return per_clip_stats, global_stats


# ----------------------------
# Main computation
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute comprehensive metrics for Driving World Model"
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        help="Optional: export metrics to CSV file"
    )
    args = parser.parse_args()
    
    print("=== Driving World Model Metrics ===\n")
    
    # Load data
    raw = pd.read_csv(RAW_PATH)
    smooth = pd.read_csv(SMOOTH_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    planner = pd.read_csv(PLANNER_PATH)
    
    # Frames per clip
    frames_per_clip = collect_frames_per_clip(FRAMES_ROOT)
    
    # ------------------------
    # 1. Clip timing & throughput
    # ------------------------
    print("=" * 70)
    print("1) CLIP TIMING & THROUGHPUT")
    print("=" * 70)
    timing_stats = compute_clip_timing(FRAMES_ROOT, VIDEOS_ROOT)
    
    global_frames = 0
    global_segments = 0
    total_time_span = 0.0
    
    for clip, stats in timing_stats.items():
        n_frames = stats['n_frames']
        n_segments = len(segments[segments["clip"] == clip])
        global_frames += n_frames
        global_segments += n_segments
        
        print(f"\n  Clip {clip}:")
        print(f"    Frames: {n_frames}")
        if stats['video_fps']:
            print(f"    Video FPS: {stats['video_fps']:.2f}")
            print(f"    Video duration: {stats['video_duration_sec']:.2f}s")
        if stats['estimated_time_span_sec']:
            total_time_span += stats['estimated_time_span_sec']
            print(f"    Estimated time span: {stats['estimated_time_span_sec']:.2f}s")
            print(f"    Frames/sec: {stats['frames_per_sec']:.2f}")
        print(f"    Segments: {n_segments}")
        if stats['estimated_time_span_sec']:
            segs_per_sec = n_segments / stats['estimated_time_span_sec'] if stats['estimated_time_span_sec'] > 0 else 0.0
            print(f"    Segments/sec: {segs_per_sec:.3f}")
    
    print("\n  Global:")
    print(f"    Total frames: {global_frames}")
    print(f"    Total segments: {global_segments}")
    if total_time_span > 0:
        print(f"    Total time span: {total_time_span:.2f}s")
        print(f"    Global frames/sec: {global_frames / total_time_span:.2f}")
        print(f"    Global segments/sec: {global_segments / total_time_span:.3f}")
    
    # ------------------------
    # 2. Frames & segments compression
    # ------------------------
    print("\n" + "=" * 70)
    print("2) PIPELINE COMPRESSION METRICS")
    print("=" * 70)
    
    global_compression = (global_frames / global_segments) if global_segments > 0 else 0.0
    n_planner_commands = len(planner)
    planner_compression = (global_segments / n_planner_commands) if n_planner_commands > 0 else 0.0
    
    print("\n  Compression ratios:")
    print(f"    Frames → Segments: {global_compression:.2f}x ({global_frames} → {global_segments})")
    print(f"    Segments → Planner commands: {planner_compression:.2f}x ({global_segments} → {n_planner_commands})")
    print(f"    Frames → Planner commands: {(global_frames / n_planner_commands) if n_planner_commands > 0 else 0.0:.2f}x ({global_frames} → {n_planner_commands})")
    
    for clip, n_frames in frames_per_clip.items():
        seg_clip = segments[segments["clip"] == clip]
        n_segments = len(seg_clip)
        compression = (n_frames / n_segments) if n_segments > 0 else 0.0
        seg_lengths = (seg_clip["end"] - seg_clip["start"] + 1).tolist()
        avg_seg = sum(seg_lengths) / len(seg_lengths) if seg_lengths else 0.0
        
        print(f"  Clip {clip}: {n_frames} frames → {n_segments} segments ({compression:.2f}x)")
        print(f"    Avg segment length: {avg_seg:.1f} frames")
    
    # ------------------------
    # 3. Segment semantics
    # ------------------------
    print("\n" + "=" * 70)
    print("3) SEGMENT SEMANTICS")
    print("=" * 70)
    
    seg_semantics_per_clip, seg_semantics_global = compute_segment_semantics(segments)
    
    print("\n  Global phase frequencies:")
    for phase, count in seg_semantics_global['phase_freq'].most_common():
        pct = 100.0 * count / sum(seg_semantics_global['phase_freq'].values()) if seg_semantics_global['phase_freq'] else 0.0
        print(f"    {phase}: {count} ({pct:.1f}%)")
    
    print("\n  Top segment transitions (global):")
    for (from_phase, to_phase), count in seg_semantics_global['transition_freq'].most_common(10):
        print(f"    {from_phase} → {to_phase}: {count}")
    
    # ------------------------
    # 4. Temporal smoothing
    # ------------------------
    print("\n" + "=" * 70)
    print("4) TEMPORAL SMOOTHING (JITTER & RUN-LENGTH)")
    print("=" * 70)
    
    fields = [
        ("affordance", "affordance_smoothed"),
        ("yield_to", "yield_to_smoothed"),
        ("lead_state", "lead_state_smoothed"),
    ]
    
    for field_raw, field_smooth in fields:
        print(f"\n  Field: {field_raw}")
        per_clip, global_stats = compute_temporal_metrics(raw, smooth, field_raw, field_smooth)
        
        for clip, stats in per_clip.items():
            print(
                f"    Clip {clip}: flips {stats['raw_flips']} → {stats['smooth_flips']}, "
                f"jitter reduction {stats['jitter_reduction_pct']:.1f}%, "
                f"mean run {stats['mean_raw_run']:.2f} → {stats['mean_smooth_run']:.2f}"
            )
        
        print(
            f"    Global: flips {global_stats['raw_flips']} → {global_stats['smooth_flips']}, "
            f"jitter reduction {global_stats['jitter_reduction_pct']:.1f}%, "
            f"mean run {global_stats['mean_raw_run']:.2f} → {global_stats['mean_smooth_run']:.2f}"
        )
    
    # ------------------------
    # 5. World-state dynamics
    # ------------------------
    print("\n" + "=" * 70)
    print("5) WORLD-STATE DYNAMICS")
    print("=" * 70)
    
    for field in ["affordance", "yield_to", "lead_state"]:
        print(f"\n  Field: {field}")
        per_clip_dynamics, global_dynamics = compute_world_state_dynamics(smooth, field)
        
        print(f"    Global entropy: {global_dynamics['entropy']:.3f}")
        print(f"    Unique states: {global_dynamics['unique_states']}")
        print(f"    Avg dwell time: {global_dynamics['avg_dwell_time']:.1f} frames")
        print(f"    Dwell range: {global_dynamics['min_dwell_time']} - {global_dynamics['max_dwell_time']} frames")
        
        print("    Top transitions:")
        trans_items = [(k, v['count']) for k, v in global_dynamics['transition_matrix'].items() if v['count'] > 0]
        trans_items.sort(key=lambda x: x[1], reverse=True)
        for (from_state, to_state), count in trans_items[:5]:
            prob = global_dynamics['transition_matrix'][(from_state, to_state)]['prob']
            print(f"      {from_state} → {to_state}: {count} ({prob*100:.1f}%)")
    
    # ------------------------
    # 6. Planner performance
    # ------------------------
    print("\n" + "=" * 70)
    print("6) PLANNER PERFORMANCE")
    print("=" * 70)
    
    planner_per_clip, planner_global = compute_planner_performance(planner)
    
    print("\n  Global command frequencies:")
    for cmd, cnt in planner_global['cmd_freq'].most_common():
        total = sum(planner_global['cmd_freq'].values())
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"    {cmd}: {cnt} ({pct:.1f}%)")
    
    print(f"\n  Unique commands: {planner_global['unique_commands']}")
    
    print("\n  Top planner transitions:")
    for (from_cmd, to_cmd), count in planner_global['transition_freq'].most_common(10):
        print(f"    {from_cmd} → {to_cmd}: {count}")
    
    for clip, stats in planner_per_clip.items():
        print(f"\n  Clip {clip}:")
        print(f"    Unique commands: {stats['unique_commands']}")
        print(f"    Avg dwell time: {stats['avg_dwell_frames']:.1f} frames")
    
    # ------------------------
    # 7. YOLO grounding (optional)
    # ------------------------
    print("\n" + "=" * 70)
    print("7) YOLO VISUAL GROUNDING")
    print("=" * 70)
    
    if YOLO_AVAILABLE:
        per_clip_yolo, global_yolo = compute_yolo_grounding(FRAMES_ROOT, segments, planner)
        
        if not per_clip_yolo:
            print("  No applicable planner rows with targets for YOLO evaluation.")
        else:
            for clip, stats in per_clip_yolo.items():
                print(f"\n  Clip {clip}: {stats['applicable']} applicable")
                print(f"    Correct: {stats['correct']} ({stats['correct_pct']:.1f}%)")
                print(f"    Wrong type: {stats['wrong_type']}")
                print(f"    None: {stats['none']}")
                print(f"    Ambiguous: {stats['ambiguous']}")
            
            if global_yolo:
                print(f"\n  Global: {global_yolo['applicable']} applicable")
                print(f"    Correct: {global_yolo['correct']} ({global_yolo['correct_pct']:.1f}%)")
                print(f"    Wrong type: {global_yolo['wrong_type']}")
                print(f"    None: {global_yolo['none']}")
                print(f"    Ambiguous: {global_yolo['ambiguous']}")
    else:
        print("  YOLO grounding skipped (YOLO not available).")
    
    # ------------------------
    # 8. Resume-friendly summary
    # ------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (RESUME EXTRACTION)")
    print("=" * 70)
    
    print("\n  Compression:")
    print(f"    • Frames → Segments: {global_compression:.2f}x compression")
    print(f"    • Frames → Planner commands: {(global_frames / n_planner_commands) if n_planner_commands > 0 else 0.0:.2f}x compression")
    
    print("\n  Temporal smoothing:")
    # Compute average jitter reduction across all fields
    total_raw_flips = 0
    total_smooth_flips = 0
    fields = [
        ("affordance", "affordance_smoothed"),
        ("yield_to", "yield_to_smoothed"),
        ("lead_state", "lead_state_smoothed"),
    ]
    for field_raw, field_smooth in fields:
        _, global_stats = compute_temporal_metrics(raw, smooth, field_raw, field_smooth)
        total_raw_flips += global_stats['raw_flips']
        total_smooth_flips += global_stats['smooth_flips']
    avg_jitter_reduction = 100.0 * (total_raw_flips - total_smooth_flips) / total_raw_flips if total_raw_flips > 0 else 0.0
    print(f"    • {avg_jitter_reduction:.1f}% jitter reduction across all fields")
    
    print("\n  Planner diversity:")
    print(f"    • {planner_global['unique_commands']} distinct planner commands")
    print(f"    • {len(planner_global['transition_freq'])} unique state transitions")
    
    if YOLO_AVAILABLE:
        per_clip_yolo, global_yolo = compute_yolo_grounding(FRAMES_ROOT, segments, planner)
        if global_yolo:
            print("\n  Visual grounding:")
            print(f"    • {global_yolo['correct_pct']:.1f}% correct target detection rate")
    
    print("\n" + "=" * 70)
    print("Done.\n")
    
    # ------------------------
    # CSV Export (if requested)
    # ------------------------
    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect key metrics for export
        export_rows = []
        
        # Compression metrics
        export_rows.append({
            'metric_category': 'compression',
            'metric_name': 'frames_to_segments',
            'value': global_compression,
            'unit': 'ratio'
        })
        export_rows.append({
            'metric_category': 'compression',
            'metric_name': 'frames_to_planner',
            'value': global_frames / n_planner_commands if n_planner_commands > 0 else 0.0,
            'unit': 'ratio'
        })
        
        # Planner metrics
        export_rows.append({
            'metric_category': 'planner',
            'metric_name': 'unique_commands',
            'value': planner_global['unique_commands'],
            'unit': 'count'
        })
        
        # YOLO metrics
        if YOLO_AVAILABLE:
            per_clip_yolo, global_yolo = compute_yolo_grounding(FRAMES_ROOT, segments, planner)
            if global_yolo:
                export_rows.append({
                    'metric_category': 'yolo',
                    'metric_name': 'correct_detection_rate',
                    'value': global_yolo['correct_pct'],
                    'unit': 'percent'
                })
        
        export_df = pd.DataFrame(export_rows)
        export_df.to_csv(export_path, index=False)
        print(f"Metrics exported to: {export_path}\n")


if __name__ == "__main__":
    main()
