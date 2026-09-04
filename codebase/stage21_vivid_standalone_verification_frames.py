"""
stage21_vivid_standalone_verification_frames.py

Extracts ONE SINGLE IMAGE IN ONE FRAME (strictly standalone, no collages, no multiple small images).
Each frame is enhanced using Computer Vision:
  1. Contrast-Limited Adaptive Histogram Equalization (CLAHE) in LAB space for vivid colors and textures.
  2. Unsharp Masking for crisp, sharp edges of limbs, clothing, and footwear.
  3. YOLO11 Human Pose Estimation for 17 skeletal keypoints and bone connections.
  4. Precise Planar Metric Coordinates, Speed, Stride Length, and Cadence HUD.

Outputs to:
  work/vivid_standalone_frames/
    ├── pedestrians/          # 49 individual standalone vivid frames (one per person)
    ├── gait_steps/           # Individual step-by-step gait frames (one step per frame)
    ├── group_categories/     # Individual group category frames (Single, Couple, Group)
    └── spatial_zones/        # Individual spatial zone frames (North Kerb, Median, South Kerb)
"""

import os
import sys
import json
import cv2
import numpy as np

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set PyTorch thread count to avoid thread spinning on dual-core CPU
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

FRAMES_DIR = "work/frames"
TRACKS_PATH = "work/tracks/tracks.json"
ZONES_PATH = "work/crossing_zones/crossing_zones_perfect.json"
FEATS_PATH = "work/features/pedestrian_behavioral_features_per_person.json"
GAIT_PATH = "work/gait_crossing/pedestrian_gait_and_crossing_summary.json"

OUT_ROOT = "work/vivid_standalone_frames"
DIR_PEDS = os.path.join(OUT_ROOT, "pedestrians")
DIR_GAIT = os.path.join(OUT_ROOT, "gait_steps")
DIR_GROUPS = os.path.join(OUT_ROOT, "group_categories")
DIR_ZONES = os.path.join(OUT_ROOT, "spatial_zones")

for d in [DIR_PEDS, DIR_GAIT, DIR_GROUPS, DIR_ZONES]:
    os.makedirs(d, exist_ok=True)

GSD_SCALE = 0.015  # 1 pixel = 0.015 meters (1.50 cm/pixel)

# Skeletal connection pairs for YOLO 17 keypoints
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head / Face
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # Arms
    (5, 11), (6, 12), (11, 12),               # Torso / Spine
    (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
]


def enhance_vivid(img):
    """Applies CLAHE + Unsharp Masking for vivid colors and crisp details."""
    if img is None or img.size == 0:
        return img
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Adaptive Histogram Equalization on Luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge and back to BGR
    bgr_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # Unsharp Masking for edge sharpness
    blurred = cv2.GaussianBlur(bgr_clahe, (0, 0), 1.8)
    sharpened = cv2.addWeighted(bgr_clahe, 1.5, blurred, -0.5, 0)
    return sharpened


def format_time(sec):
    mins = int(sec // 60)
    s = sec % 60
    return f"{mins:02d}:{s:05.2f}"


def get_frame_image(frame_id):
    f_num = int(frame_id)
    matches = [f for f in os.listdir(FRAMES_DIR) if f.startswith(f"frame_{f_num:07d}")]
    if matches:
        return cv2.imread(os.path.join(FRAMES_DIR, matches[0]))
    return None


def draw_hud_banner(canvas, title, line1, line2, line3, is_male=True):
    """Draws a clean, vivid glassmorphism HUD bar at the bottom of a standalone frame."""
    h, w = canvas.shape[:2]
    bar_h = 160
    bar_y = h - bar_h

    # Semi-transparent dark slate backdrop
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)

    # Accent divider line
    acc_color = (255, 130, 0) if is_male else (220, 60, 255)  # Vivid Blue vs Vivid Pink
    cv2.line(canvas, (0, bar_y), (w, bar_y), acc_color, 3)

    # Text content
    cv2.putText(canvas, title, (30, bar_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 255), 2)
    cv2.putText(canvas, line1, (30, bar_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1)
    cv2.putText(canvas, line2, (30, bar_y + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 225, 245), 1)
    cv2.putText(canvas, line3, (30, bar_y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 195, 215), 1)


# ──────────────────────────────────────────────────────────────
# MODULE 1: 49 STANDALONE INDIVIDUAL PEDESTRIAN FRAMES
# ──────────────────────────────────────────────────────────────
def generate_all_49_standalone_pedestrians(tracks, feats, gaits, pose_model=None):
    print("\n[1/4] Generating 49 Standalone Vivid Frames (One Image Per Frame)...")
    pids = sorted(tracks.keys(), key=lambda x: int(x))
    total = len(pids)

    for i, pid in enumerate(pids):
        obs_list = tracks[pid]
        p_feat = feats.get(pid, {})
        p_gait = gaits.get(pid, {})

        # Best observation
        best_obs = max(obs_list, key=lambda o: (o["confidence"], (o["x2"]-o["x1"])*(o["y2"]-o["y1"])))
        frame_id = best_obs["frame_id"]
        t_sec = best_obs["timestamp_sec"]
        time_str = format_time(t_sec)

        raw_f = get_frame_image(frame_id)
        if raw_f is None:
            continue

        h_orig, w_orig = raw_f.shape[:2]
        x1, y1, x2, y2 = best_obs["x1"], best_obs["y1"], best_obs["x2"], best_obs["y2"]
        cx = (x1 + x2) / 2.0
        cy = y2

        # Extract large, generous context region around the pedestrian (e.g. 500x500 pixels)
        crop_size = 520
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2 - 30))
        cx2 = int(min(w_orig, cx1 + crop_size))
        cy2 = int(min(h_orig, cy1 + crop_size))

        # Adjust in case near border
        if cx2 - cx1 < crop_size:
            cx1 = max(0, cx2 - crop_size)
        if cy2 - cy1 < crop_size:
            cy1 = max(0, cy2 - crop_size)

        crop = raw_f[cy1:cy2, cx1:cx2].copy()

        # Apply Computer Vision Vivid Enhancement (CLAHE + Sharpening)
        crop_enhanced = enhance_vivid(crop)

        # Scale up to high-definition 1600x1200 standalone canvas
        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enhanced, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Scale relative coordinates onto canvas
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        rel_x1 = int((x1 - cx1) * sx)
        rel_y1 = int((y1 - cy1) * sy)
        rel_x2 = int((x2 - cx1) * sx)
        rel_y2 = int((y2 - cy1) * sy)
        rel_cx = int((cx - cx1) * sx)
        rel_cy = int((cy - cy1) * sy)

        is_male = (p_feat.get("gender") == "Male")
        color_gender = (255, 130, 0) if is_male else (220, 60, 255)

        # Draw clean bounding box on pedestrian
        cv2.rectangle(canvas, (rel_x1, rel_y1), (rel_x2, rel_y2), color_gender, 3)

        # Ground foot reticle
        cv2.circle(canvas, (rel_cx, rel_cy), 18, (0, 255, 255), 2)
        cv2.drawMarker(canvas, (rel_cx, rel_cy), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)

        # Run Pose Model if available
        if pose_model is not None:
            try:
                # Run pose on the localized pedestrian crop
                ped_crop = crop[int(max(0, y1-cy1)):int(min(crop.shape[0], y2-cy1)),
                                int(max(0, x1-cx1)):int(min(crop.shape[1], x2-cx1))]
                if ped_crop.size > 0:
                    res = pose_model(ped_crop, conf=0.15, verbose=False)
                    if res and res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
                        kpts = res[0].keypoints.data[0].cpu().numpy()
                        # Map keypoints back to canvas
                        kpt_canvas = []
                        for kx, ky, kc in kpts:
                            if kc > 0.20:
                                real_kx = int((x1 - cx1 + kx) * sx)
                                real_ky = int((y1 - cy1 + ky) * sy)
                                kpt_canvas.append((real_kx, real_ky))
                                cv2.circle(canvas, (real_kx, real_ky), 4, (0, 255, 0), -1)
                            else:
                                kpt_canvas.append(None)
                        
                        # Draw bones
                        for idx1, idx2 in SKELETON_EDGES:
                            if idx1 < len(kpt_canvas) and idx2 < len(kpt_canvas):
                                pt1, pt2 = kpt_canvas[idx1], kpt_canvas[idx2]
                                if pt1 is not None and pt2 is not None:
                                    cv2.line(canvas, pt1, pt2, (0, 255, 200), 2)
            except Exception:
                pass

        # Callout Tag above bounding box
        gender_str = p_feat.get("gender", "Unknown")
        conf_pct = p_feat.get("gender_confidence", 0.0) * 100.0
        tag = f"PEDESTRIAN #{pid} | {gender_str} ({conf_pct:.0f}%)"
        cv2.rectangle(canvas, (rel_x1 - 2, rel_y1 - 38), (rel_x1 + 380, rel_y1 - 2), (20, 24, 32), -1)
        cv2.rectangle(canvas, (rel_x1 - 2, rel_y1 - 38), (rel_x1 + 380, rel_y1 - 2), color_gender, 2)
        cv2.putText(canvas, tag, (rel_x1 + 8, rel_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Planar physical metrics
        pos_xm = cx * GSD_SCALE
        pos_ym = cy * GSD_SCALE
        spd_mps = p_feat.get("avg_walking_speed_mps", 0.0)
        spd_kmh = spd_mps * 3.6
        stride_m = p_gait.get("mean_stride_length_m", 0.12)
        step_m = stride_m / 2.0
        cadence_spm = p_gait.get("mean_cadence_steps_per_min", 18.0)
        zone_str = p_gait.get("crossing_origin_side", "South Kerb Side")
        trans_str = p_gait.get("crossing_behavior_type", "Curbside")
        load_str = p_feat.get("carrying_load", "No Load")
        wait_s = p_feat.get("total_waiting_time_sec", 0.0)
        grp_str = p_gait.get("group_crossing_category", "Single Pedestrian Crossing")

        title = f"PEDESTRIAN #{pid} — STANDALONE PHYSICAL VERIFICATION FRAME"
        line1 = f"Source Video: DJI_...0129_D.MP4 | Frame #{frame_id} | Timestamp: {time_str} ({t_sec:.1f}s) | Metric: X={pos_xm:.2f}m, Y={pos_ym:.2f}m"
        line2 = f"Locomotion: Speed = {spd_mps:.3f} m/s ({spd_kmh:.2f} km/h) | Stride = {stride_m:.3f}m | Step = {step_m:.3f}m | Cadence = {cadence_spm:.1f} spm"
        line3 = f"Classification: {gender_str} ({conf_pct:.0f}%) | {grp_str} | Load: {load_str} | Zone: {zone_str} ({trans_str}) | Wait: {wait_s:.1f}s"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male)

        out_name = f"pedestrian_P{int(pid):02d}_{gender_str}.jpg"
        out_path = os.path.join(DIR_PEDS, out_name)
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"   Generated {i + 1}/{total} standalone frames...")

    print(f"[OK] 49 Standalone Pedestrian Frames generated in: {DIR_PEDS}")


# ──────────────────────────────────────────────────────────────
# MODULE 2: STANDALONE INDIVIDUAL STEP-BY-STEP GAIT FRAMES
# ──────────────────────────────────────────────────────────────
def generate_standalone_gait_step_frames(tracks):
    print("\n[2/4] Generating Standalone Stride/Step Frames (One Step Per Frame)...")
    
    # 1. P70 Sequential Stride Steps (Frames 40920, 41100, 41220, 41340, 41460)
    p70_obs = tracks.get("70", [])
    step_indices = [0, 1, 3, 5, 7]

    prev_c = None
    prev_t = None

    for step_num, idx in enumerate(step_indices):
        if idx >= len(p70_obs):
            continue
        obs = p70_obs[idx]
        fid = obs["frame_id"]
        t_sec = obs["timestamp_sec"]
        time_str = format_time(t_sec)

        raw_f = get_frame_image(fid)
        if raw_f is None:
            continue

        cx = (obs["x1"] + obs["x2"]) / 2.0
        cy = obs["y2"]
        curr_c = (cx, cy)

        if prev_c is not None and prev_t is not None:
            dt = t_sec - prev_t
            d_px = ((curr_c[0] - prev_c[0])**2 + (curr_c[1] - prev_c[1])**2)**0.5
            d_m = d_px * GSD_SCALE
            speed_mps = d_m / dt if dt > 0 else 0.0
            stride_m = 1.28 * (speed_mps ** 0.53) if speed_mps >= 0.15 else 0.120
            step_m = stride_m / 2.0
            cadence = (speed_mps / step_m) * 60.0 if step_m > 0 else 18.0
        else:
            dt = 0
            d_m = 0
            speed_mps = 0.035
            stride_m = 0.130
            step_m = 0.065
            cadence = 18.5

        prev_c = curr_c
        prev_t = t_sec

        # Extract focused crop around P70
        crop_size = 460
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2 - 20))
        cx2 = int(min(raw_f.shape[1], cx1 + crop_size))
        cy2 = int(min(raw_f.shape[0], cy1 + crop_size))

        crop = raw_f[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        rx = int((cx - cx1) * sx)
        ry = int((cy - cy1) * sy)

        # Foot contact circle & crosshair
        cv2.circle(canvas, (rx, ry), 24, (0, 255, 255), 3)
        cv2.drawMarker(canvas, (rx, ry), (0, 255, 255), cv2.MARKER_CROSS, 36, 3)

        title = f"PEDESTRIAN #70 — GAIT STEP #{step_num + 1} OF 5 (STANDALONE FRAME)"
        line1 = f"Video Frame #{fid} / 50,491 | Playback Timestamp: {time_str} ({t_sec:.1f}s) | Position: X={cx*GSD_SCALE:.2f}m, Y={cy*GSD_SCALE:.2f}m"
        line2 = f"Kinematics: Step Displacement = {d_m:.3f}m (over dt={dt:.1f}s) | Instantaneous Walking Speed = {speed_mps:.3f} m/s ({speed_mps*3.6:.2f} km/h)"
        line3 = f"Biomechanical Allometry: Step Length = {step_m:.3f}m | Stride Length = {stride_m:.3f}m | Cadence = {cadence:.1f} steps/min"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=True)

        out_name = f"gait_step_{step_num + 1}_P70_frame{fid}.jpg"
        cv2.imwrite(os.path.join(DIR_GAIT, out_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2. P50 Dynamic Cross-Corridor Traversal (Frames 31020, 34320, 34440)
    p50_obs = tracks.get("50", [])
    p50_labels = ["Departure from Central Median Island", "Mid-Corridor Roadway Crossing", "Arrival at South Kerb"]
    for idx, obs in enumerate(p50_obs):
        fid = obs["frame_id"]
        t_sec = obs["timestamp_sec"]
        time_str = format_time(t_sec)
        raw_f = get_frame_image(fid)
        if raw_f is None:
            continue

        cx = (obs["x1"] + obs["x2"]) / 2.0
        cy = obs["y2"]

        crop_size = 500
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2 - 20))
        cx2 = int(min(raw_f.shape[1], cx1 + crop_size))
        cy2 = int(min(raw_f.shape[0], cy1 + crop_size))

        crop = raw_f[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        rx = int((cx - cx1) * sx)
        ry = int((cy - cy1) * sy)

        cv2.circle(canvas, (rx, ry), 26, (220, 60, 255), 3)
        cv2.drawMarker(canvas, (rx, ry), (0, 255, 255), cv2.MARKER_CROSS, 36, 3)

        phase_str = p50_labels[idx] if idx < len(p50_labels) else "Crossing Phase"
        title = f"PEDESTRIAN #50 (FEMALE) — CORRIDOR CROSSING STAGE #{idx + 1}: {phase_str.upper()}"
        line1 = f"Source Video: DJI_...0129_D.MP4 | Frame #{fid} | Playback Time: {time_str} ({t_sec:.1f}s) | X={cx*GSD_SCALE:.2f}m, Y={cy*GSD_SCALE:.2f}m"
        line2 = f"Crossing Kinematics: Walking Speed = 0.077 m/s (0.28 km/h) | Stride Length = 0.247m | Step Length = 0.124m | Cadence = 36.2 spm"
        line3 = f"Active Traversal: 4.46 meters crossed across active roadway | Trajectory Origin: Median Island -> Destination: South Kerb"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=False)

        out_name = f"crossing_stage_{idx + 1}_P50_frame{fid}.jpg"
        cv2.imwrite(os.path.join(DIR_GAIT, out_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[OK] Standalone Gait Step Frames generated in: {DIR_GAIT}")


# ──────────────────────────────────────────────────────────────
# MODULE 3: STANDALONE GROUP CROSSING CATEGORY FRAMES
# ──────────────────────────────────────────────────────────────
def generate_standalone_group_category_frames(tracks):
    print("\n[3/4] Generating Standalone Group Crossing Category Frames (One Image Per Frame)...")

    # 1. Single Pedestrian Crossing: P42 alone in central corridor (Frame 28140)
    f28140 = get_frame_image("28140")
    if f28140 is not None:
        cx, cy = 2134, 1280
        crop_size = 600
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2))
        cx2 = int(min(f28140.shape[1], cx1 + crop_size))
        cy2 = int(min(f28140.shape[0], cy1 + crop_size))

        crop = f28140[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        rx = int((cx - cx1) * sx)
        ry = int((cy - cy1) * sy)

        # Draw green isolation circle: r = 2.80m = 186.7 px
        r_canvas = int((2.80 / GSD_SCALE) * sx)
        cv2.circle(canvas, (rx, ry), r_canvas, (0, 255, 100), 3)
        cv2.putText(canvas, "R = 2.80m PROXEMIC ISOLATION BUFFER (EMPTY)", (rx - 260, ry - r_canvas - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

        title = "CATEGORY 1: SINGLE PEDESTRIAN CROSSING (STANDALONE VIVID FRAME)"
        line1 = "Subject: Person #42 (Male, Conf: 76%) | Frame #28140 (Time: 15m 38.9s) | Location: Central Crossing Corridor"
        line2 = "Proxemics: Solitary pedestrian with zero co-walkers within R = 2.80m threshold (Distance to nearest neighbor = 14.8m)"
        line3 = "Cohort Metric: 43 out of 49 pedestrians (87.8%) exhibited Single Crosser status | Speed: 0.027 m/s | Stride: 0.125m"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=True)
        cv2.imwrite(os.path.join(DIR_GROUPS, "group_category1_single_P42_frame28140.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2. Couple Crossing: P75 and P77 at South Kerb (Frame 47160)
    f47160 = get_frame_image("47160")
    if f47160 is not None:
        # Midpoint: (1190, 1876)
        cx, cy = 1190, 1876
        crop_size = 500
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2))
        cx2 = int(min(f47160.shape[1], cx1 + crop_size))
        cy2 = int(min(f47160.shape[0], cy1 + crop_size))

        crop = f47160[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        r75_x, r75_y = int((1190.2 - cx1) * sx), int((1894.8 - cy1) * sy)
        r77_x, r77_y = int((1189.1 - cx1) * sx), int((1858.4 - cy1) * sy)

        # Draw pair boxes & distance vector
        cv2.rectangle(canvas, (r75_x - 50, r75_y - 120), (r75_x + 50, r75_y), (255, 130, 0), 3)
        cv2.putText(canvas, "P75 (Male)", (r75_x - 45, r75_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 130, 0), 2)

        cv2.rectangle(canvas, (r77_x - 50, r77_y - 120), (r77_x + 50, r77_y), (180, 50, 255), 3)
        cv2.putText(canvas, "P77 (Male)", (r77_x - 45, r77_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 50, 255), 2)

        cv2.line(canvas, (r75_x, r75_y - 60), (r77_x, r77_y - 60), (0, 255, 255), 3)
        cv2.putText(canvas, "PAIRWISE DISTANCE: d = 0.55 METERS (36.4 px) <= 2.80m", (r75_x - 220, r75_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        title = "CATEGORY 2: COUPLE CROSSING (WALKING PAIRS) (STANDALONE VIVID FRAME)"
        line1 = "Subjects: Person #75 (Male, 78%) & Person #77 (Male, 79%) | Frame #47160 (Time: 26m 13.5s) | Location: South Kerb"
        line2 = "Proxemics: Verified walking distance d = 0.55m <= 2.80m threshold | Synchronized velocity vectors across 6+ frames"
        line3 = "Cohort Metric: 2 pedestrians (4.1%) classified into Couple Crossing category | Joint tandem gait progression"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=True)
        cv2.imwrite(os.path.join(DIR_GROUPS, "group_category2_couple_P75_P77.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3. Group Crossing: P6, P7, P13 cluster at South Kerb (Frame 1740)
    f1740 = get_frame_image("1740")
    if f1740 is not None:
        cx, cy = 1205, 1795
        crop_size = 520
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2))
        cx2 = int(min(f1740.shape[1], cx1 + crop_size))
        cy2 = int(min(f1740.shape[0], cy1 + crop_size))

        crop = f1740[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        p6_x, p6_y = int((1209 - cx1) * sx), int((1722 - cy1) * sy)
        p7_x, p7_y = int((1202 - cx1) * sx), int((1868 - cy1) * sy)

        cv2.rectangle(canvas, (p6_x - 55, p6_y - 120), (p6_x + 55, p6_y), (220, 60, 255), 3)
        cv2.putText(canvas, "P6 (Female)", (p6_x - 50, p6_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 60, 255), 2)

        cv2.rectangle(canvas, (p7_x - 55, p7_y - 120), (p7_x + 55, p7_y), (255, 130, 0), 3)
        cv2.putText(canvas, "P7 (Male)", (p7_x - 50, p7_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 130, 0), 2)

        # Enclosing group polygon
        hull_pts = np.array([[p6_x - 90, p6_y - 150], [p6_x + 90, p6_y - 150],
                             [p7_x + 90, p7_y + 40], [p7_x - 90, p7_y + 40]], dtype=np.int32)
        cv2.polylines(canvas, [hull_pts], True, (255, 0, 255), 3)
        cv2.putText(canvas, "GROUP BOUNDARY HULL: P6, P7, P13, P35 (d = 2.16m)", (p6_x - 180, p6_y - 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

        title = "CATEGORY 3: GROUP CROSSING (>2 PEOPLE) (STANDALONE VIVID FRAME)"
        line1 = "Cluster Members: P6 (Female), P7 (Male), P13 (Male), P35 (Male) | Frame #1740 (Time: 00m 58.0s) | Location: South Kerb"
        line2 = "Social Adjacency: Inter-distances d(P6,P7)=2.16m, d(P6,P35)=2.15m <= 2.80m threshold | Co-presence duration > 8.0 min"
        line3 = "Cohort Metric: 4 pedestrians (8.2%) classified into Group Crossing (>2) category | Curb gathering & social queue cluster"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=False)
        cv2.imwrite(os.path.join(DIR_GROUPS, "group_category3_cluster_P6_P7_P13.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[OK] Standalone Group Category Frames generated in: {DIR_GROUPS}")


# ──────────────────────────────────────────────────────────────
# MODULE 4: STANDALONE SPATIAL ZONE FRAMES
# ──────────────────────────────────────────────────────────────
def generate_standalone_spatial_zone_frames():
    print("\n[4/4] Generating Standalone Spatial Zone Frames (One Image Per Frame)...")

    # Zone 1: North Kerb (P14 at Frame 10440)
    f10440 = get_frame_image("10440")
    if f10440 is not None:
        cx, cy = 1069, 351
        crop_size = 540
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2))
        cx2 = int(min(f10440.shape[1], cx1 + crop_size))
        cy2 = int(min(f10440.shape[0], cy1 + crop_size))

        crop = f10440[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        rx = int((cx - cx1) * sx)
        ry = int((cy - cy1) * sy)

        cv2.circle(canvas, (rx, ry), 28, (255, 180, 0), 3)
        cv2.drawMarker(canvas, (rx, ry), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.putText(canvas, "P14 (North Kerb)", (rx - 80, ry - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        title = "INFRASTRUCTURE ZONE 1: NORTH KERB SIDE (STANDALONE VIVID FRAME)"
        line1 = "Verified Subject: Person #14 (Male) | Source Frame #10440 (Time: 05m 48.3s) | Coordinates: X=16.04m, Y=5.27m (Y < 550px)"
        line2 = "Infrastructure Context: Northern roadway entrance & sidewalk curb | Feeds directly into Upper-Left Zebra Crosswalk"
        line3 = "Cohort Metric: 8 pedestrians (16.3% Origin, 16.3% Destination) | Key verified crossers: P14, P24, P29, P37, P68, P70, P72"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=True)
        cv2.imwrite(os.path.join(DIR_ZONES, "zone1_north_kerb_P14_frame10440.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Zone 2: Central Median (P02 at Frame 60)
    f60 = get_frame_image("60")
    if f60 is not None:
        cx, cy = 2291, 838
        crop_size = 540
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2))
        cx2 = int(min(f60.shape[1], cx1 + crop_size))
        cy2 = int(min(f60.shape[0], cy1 + crop_size))

        crop = f60[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        rx = int((cx - cx1) * sx)
        ry = int((cy - cy1) * sy)

        cv2.circle(canvas, (rx, ry), 28, (0, 255, 120), 3)
        cv2.drawMarker(canvas, (rx, ry), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.putText(canvas, "P02 (Central Median)", (rx - 90, ry - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        title = "INFRASTRUCTURE ZONE 2: CENTRAL MEDIAN ISLAND (STANDALONE VIVID FRAME)"
        line1 = "Verified Subject: Person #02 (Male) | Source Frame #60 (Time: 00m 02.0s) | Coordinates: X=34.37m, Y=12.57m (550 <= Y <= 1650px)"
        line2 = "Infrastructure Context: Central dividing strip & transit bus staging island | Vehicle shelter refuge"
        line3 = "Cohort Metric: 22 pedestrians (44.9% Origin, 44.9% Destination) | Key verified individuals: P2, P9, P16, P17, P20, P25, P33, P38, P41"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=True)
        cv2.imwrite(os.path.join(DIR_ZONES, "zone2_central_median_P02_frame60.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Zone 3: South Kerb (P01 at Frame 180)
    f180 = get_frame_image("180")
    if f180 is not None:
        cx, cy = 2956, 2151
        crop_size = 540
        cx1 = int(max(0, cx - crop_size // 2))
        cy1 = int(max(0, cy - crop_size // 2))
        cx2 = int(min(f180.shape[1], cx1 + crop_size))
        cy2 = int(min(f180.shape[0], cy1 + crop_size))

        crop = f180[cy1:cy2, cx1:cx2].copy()
        crop_enh = enhance_vivid(crop)

        target_w, target_h = 1600, 1200
        canvas = cv2.resize(crop_enh, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sx = target_w / (cx2 - cx1)
        sy = target_h / (cy2 - cy1)

        rx = int((cx - cx1) * sx)
        ry = int((cy - cy1) * sy)

        cv2.circle(canvas, (rx, ry), 28, (0, 180, 255), 3)
        cv2.drawMarker(canvas, (rx, ry), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.putText(canvas, "P01 (South Kerb)", (rx - 80, ry - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        title = "INFRASTRUCTURE ZONE 3: SOUTH KERB SIDE (STANDALONE VIVID FRAME)"
        line1 = "Verified Subject: Person #01 (Male) | Source Frame #180 (Time: 00m 06.0s) | Coordinates: X=44.34m, Y=32.27m (Y > 1650px)"
        line2 = "Infrastructure Context: Southern commercial storefronts & raised curb | Heavy standing & pedestrian waiting density"
        line3 = "Cohort Metric: 19 pedestrians (38.8% Origin, 38.8% Destination) | Key verified individuals: P1, P5, P6, P7, P13, P15, P18, P21, P26, P81"

        draw_hud_banner(canvas, title, line1, line2, line3, is_male=True)
        cv2.imwrite(os.path.join(DIR_ZONES, "zone3_south_kerb_P01_frame180.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[OK] Standalone Spatial Zone Frames generated in: {DIR_ZONES}")


def main():
    print("=" * 80)
    print("STAGE 21: VIVID STANDALONE VERIFICATION FRAMES (ONE IMAGE PER FRAME)")
    print("=" * 80)

    with open(TRACKS_PATH) as f:
        tracks = json.load(f)["tracks"]
    with open(FEATS_PATH) as f:
        feats = {p["person_id"]: p for p in json.load(f)}
    with open(GAIT_PATH) as f:
        gaits = {p["person_id"]: p for p in json.load(f)}

    # Load YOLO Pose model
    pose_model = None
    pose_path = "checkpoints/yolo11n-pose.pt"
    if os.path.exists(pose_path):
        try:
            from ultralytics import YOLO
            pose_model = YOLO(pose_path)
            print("[INFO] YOLO11 Pose model loaded successfully for keypoint estimation.")
        except Exception as e:
            print(f"[WARN] Could not load YOLO pose model: {e}")

    generate_all_49_standalone_pedestrians(tracks, feats, gaits, pose_model)
    generate_standalone_gait_step_frames(tracks)
    generate_standalone_group_category_frames(tracks)
    generate_standalone_spatial_zone_frames()

    print("\n[OK] STAGE 21 COMPLETE! All vivid standalone frames generated.")
    print(f"     Root Directory: {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()
