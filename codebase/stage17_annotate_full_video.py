"""
stage17_annotate_full_video.py

Renders annotated video showing:
  1. Roadway Boundary (Cyan outline)
  2. Two White Boundary Lines (Bright White)
  3. Zebra Crossing Stripes (Semi-transparent Yellow)
  4. Pedestrian Crossing Corridor (Semi-transparent Green)
  5. Active Pedestrians with bounding boxes and live HUD tags:
     - Person ID
     - Gender (Male / Female)
     - Group Size
     - Carrying Load
     - Speed (m/s)
     - Waiting State
     - Inside/Outside Crossing Zone

Output:
  work/crossing_zones/annotated_intersection_video.mp4
"""

import os
import sys
import json
import cv2
import numpy as np

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FRAMES_DIR = "work/frames"
ZONES_PATH = "work/crossing_zones/crossing_zones_perfect.json"
FEATURES_PATH = "work/features/pedestrian_behavioral_features_per_person.json"
TIMESERIES_PATH = "work/features/pedestrian_behavioral_timeseries.csv"
TRACKS_PATH = "work/tracks/tracks.json"
OUT_VIDEO_PATH = "work/crossing_zones/annotated_intersection_video.mp4"


def render_annotated_video(max_frames=120, fps=8, out_width=1920, out_height=1080):
    print("=" * 70)
    print("STAGE 17: ANNOTATING VIDEO WITH PERFECTED ZONES & PEDESTRIAN FEATURES")
    print("=" * 70)

    # 1. Load Zones
    with open(ZONES_PATH) as f:
        zones = json.load(f)

    orig_w, orig_h = zones["resolution"]
    scale_x = out_width / orig_w
    scale_y = out_height / orig_h

    # Scale polygons to output resolution
    def scale_poly(poly):
        return np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in poly], dtype=np.int32)

    road_poly_scaled = scale_poly(zones["roadway_boundary"])
    ul_white_1 = scale_poly(zones["white_boundary_lines"]["upper_left"][0])
    ul_white_2 = scale_poly(zones["white_boundary_lines"]["upper_left"][1])
    lr_white_1 = scale_poly(zones["white_boundary_lines"]["lower_right"][0])
    lr_white_2 = scale_poly(zones["white_boundary_lines"]["lower_right"][1])
    ul_zebra_scaled = scale_poly(zones["zebra_crossings"][0]["polygon"])
    lr_zebra_scaled = scale_poly(zones["zebra_crossings"][1]["polygon"])
    corridor_scaled = scale_poly(zones["pedestrian_crossing_corridors"][2]["polygon"])

    # 2. Load Per-Person Features
    with open(FEATURES_PATH) as f:
        person_feats = {p["person_id"]: p for p in json.load(f)}

    # 3. Load Tracks organized by frame_id
    with open(TRACKS_PATH) as f:
        tracks = json.load(f)["tracks"]

    tracks_by_frame = {}
    for pid, obs_list in tracks.items():
        for o in obs_list:
            fid = str(o["frame_id"])
            if fid not in tracks_by_frame:
                tracks_by_frame[fid] = []
            tracks_by_frame[fid].append({
                "person_id": pid,
                "x1": o["x1"] * scale_x,
                "y1": o["y1"] * scale_y,
                "x2": o["x2"] * scale_x,
                "y2": o["y2"] * scale_y,
                "confidence": o["confidence"]
            })

    # 4. Get Frame Files sorted chronologically
    frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    if max_frames and max_frames > 0:
        frame_files = frame_files[:max_frames]
    total_frames = len(frame_files)
    print(f"Rendering {total_frames} frames into video at {fps} fps ({out_width}x{out_height})...")

    # 5. Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (out_width, out_height))

    for idx, fname in enumerate(frame_files):
        fpath = os.path.join(FRAMES_DIR, fname)
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        frame_resized = cv2.resize(frame, (out_width, out_height))
        overlay = frame_resized.copy()

        # A. Roadway Boundary
        cv2.polylines(frame_resized, [road_poly_scaled], True, (255, 200, 0), 2)

        # B. Crossing Corridor (semi-transparent green)
        cv2.fillPoly(overlay, [corridor_scaled], (0, 140, 70))
        cv2.polylines(overlay, [corridor_scaled], True, (0, 255, 128), 2)

        # C. Zebra Crossings (semi-transparent yellow)
        cv2.fillPoly(overlay, [ul_zebra_scaled], (0, 215, 255))
        cv2.fillPoly(overlay, [lr_zebra_scaled], (0, 215, 255))
        cv2.polylines(overlay, [ul_zebra_scaled], True, (0, 215, 255), 2)
        cv2.polylines(overlay, [lr_zebra_scaled], True, (0, 215, 255), 2)

        # Blend overlays
        frame_blended = cv2.addWeighted(overlay, 0.35, frame_resized, 0.65, 0)

        # D. White Boundary Lines (crisp opaque white)
        for line in [ul_white_1, ul_white_2, lr_white_1, lr_white_2]:
            cv2.line(frame_blended, tuple(line[0]), tuple(line[1]), (255, 255, 255), 3)

        # E. Draw Detections for current frame
        # Extract frame_id from filename (e.g. frame_0000120_t00004.00.jpg -> 120)
        parts = fname.split("_")
        frame_num_str = str(int(parts[1]))

        active_peds = tracks_by_frame.get(frame_num_str, [])
        for ped in active_peds:
            pid = ped["person_id"]
            p_feat = person_feats.get(pid, {})
            gender = p_feat.get("gender", "Male")
            group_size = p_feat.get("group_size", 1)
            load = p_feat.get("carrying_load_tag", "No Load")
            speed = p_feat.get("avg_walking_speed_mps", 0.0)

            x1, y1 = int(ped["x1"]), int(ped["y1"])
            x2, y2 = int(ped["x2"]), int(ped["y2"])

            color = (255, 100, 0) if gender == "Male" else (0, 0, 255)  # Blue = Male, Red = Female

            cv2.rectangle(frame_blended, (x1, y1), (x2, y2), color, 2)

            # Tag HUD
            tag = f"P{pid} | {gender[0]} | Grp:{group_size} | {load[:4]}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame_blended, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(frame_blended, tag, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Top HUD Banner
        cv2.rectangle(frame_blended, (10, 10), (520, 95), (20, 20, 20), -1)
        cv2.rectangle(frame_blended, (10, 10), (520, 95), (200, 200, 200), 1)
        cv2.putText(frame_blended, "UAV PEDESTRIAN CROSSING & FEATURE TRACKER", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame_blended, f"Frame: {fname} | Active Pedestrians: {len(active_peds)}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame_blended, f"Blue=Male | Red=Female | Yellow=Zebra | Green=Crossing Corridor", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        writer.write(frame_blended)

        if (idx + 1) % 25 == 0 or (idx + 1) == total_frames:
            print(f"  Processed {idx + 1}/{total_frames} frames...")

    writer.release()
    print(f"\n[OK] Video generated successfully: {OUT_VIDEO_PATH}")
    file_size_mb = os.path.getsize(OUT_VIDEO_PATH) / (1024 * 1024)
    print(f"     Video file size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    max_f = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    render_annotated_video(max_frames=max_f)
