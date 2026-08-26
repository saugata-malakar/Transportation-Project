"""
stage4_tracking.py
Section 5 — ByteTrack Tracking

Loads detections from stage3.
Runs ByteTrack to associate detections across frames.
Filters short tracks and outputs tracks.json.
"""

import argparse
import json
import os
import numpy as np

try:
    import supervision as sv
except ImportError:
    print("Please install supervision: pip install supervision")
    import sys
    sys.exit(1)

from config import TRACKING

def load_detections(det_path):
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Detections file not found: {det_path}")
    with open(det_path, "r") as f:
        return json.load(f)

def load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        return json.load(f)

def run_tracking(detections_data, manifest_data):
    try:
        tracker = sv.ByteTrack(
            track_activation_threshold=TRACKING.track_thresh,
            minimum_matching_threshold=TRACKING.match_thresh,
            lost_track_buffer=TRACKING.track_buffer,
        )
    except TypeError:
        try:
            tracker = sv.ByteTrack(
                track_thresh=TRACKING.track_thresh,
                match_thresh=TRACKING.match_thresh,
                track_buffer=TRACKING.track_buffer,
            )
        except Exception:
            tracker = sv.ByteTrack()

    frames = sorted(detections_data.get("frames", []), key=lambda x: x["frame_idx"])
    
    frame_idx_to_ts = {
        frame["frame_idx"]: frame.get("timestamp_sec", 0.0)
        for frame in frames
    }

    raw_tracks = {}

    for frame_data in frames:
        frame_idx = frame_data["frame_idx"]
        ts = frame_idx_to_ts.get(frame_idx, 0.0)
        dets = frame_data.get("detections", [])
        
        if not dets:
            tracker.update_with_detections(sv.Detections.empty())
            continue

        xyxy = []
        confidence = []
        class_id = []
        
        for d in dets:
            if d.get("class_name") != "person" and d.get("class_id") != 0:
                continue
            xyxy.append([d["x1"], d["y1"], d["x2"], d["y2"]])
            confidence.append(d["confidence"])
            class_id.append(d.get("class_id", 0))

        if not xyxy:
            tracker.update_with_detections(sv.Detections.empty())
            continue
            
        sv_dets = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )
        
        tracked_dets = tracker.update_with_detections(sv_dets)
        
        for i in range(len(tracked_dets)):
            t_xyxy = tracked_dets.xyxy[i]
            t_conf = tracked_dets.confidence[i] if tracked_dets.confidence is not None else 1.0
            t_id = tracked_dets.tracker_id[i]
            
            str_id = str(t_id)
            if str_id not in raw_tracks:
                raw_tracks[str_id] = []
                
            raw_tracks[str_id].append({
                "frame_id": str(frame_idx),
                "timestamp_sec": ts,
                "x1": float(t_xyxy[0]),
                "y1": float(t_xyxy[1]),
                "x2": float(t_xyxy[2]),
                "y2": float(t_xyxy[3]),
                "confidence": float(t_conf)
            })

    # Filter tracks shorter than min_track_len
    filtered_tracks = {}
    filtered_out = 0
    total_len = 0
    max_len = 0
    
    for t_id, track_list in raw_tracks.items():
        t_len = len(track_list)
        if t_len >= TRACKING.min_track_len:
            filtered_tracks[t_id] = track_list
            total_len += t_len
            if t_len > max_len:
                max_len = t_len
        else:
            filtered_out += 1

    total_kept = len(filtered_tracks)
    mean_len = total_len / total_kept if total_kept > 0 else 0

    print("Tracking Statistics:")
    print(f"  Total tracks initially: {len(raw_tracks)}")
    print(f"  Tracks filtered out (<{TRACKING.min_track_len} len): {filtered_out}")
    print(f"  Total tracks kept: {total_kept}")
    print(f"  Mean track length: {mean_len:.2f}")
    print(f"  Max track length: {max_len}")
    
    return {"tracks": filtered_tracks}

def main():
    parser = argparse.ArgumentParser(description="Section 5 - ByteTrack Tracking")
    parser.add_argument("--detections", required=True, help="Path to detections.json")
    parser.add_argument("--manifest", required=True, help="Path to frames_manifest.json")
    parser.add_argument("--out_dir", required=True, help="Output directory for tracks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading detections from {args.detections}")
    detections_data = load_detections(args.detections)
    
    print(f"Loading manifest from {args.manifest}")
    manifest_data = load_manifest(args.manifest)
    
    tracks = run_tracking(detections_data, manifest_data)
    
    out_file = os.path.join(args.out_dir, "tracks.json")
    with open(out_file, "w") as f:
        json.dump(tracks, f, indent=2)
        
    print(f"Tracks saved to {out_file}")

if __name__ == "__main__":
    main()
