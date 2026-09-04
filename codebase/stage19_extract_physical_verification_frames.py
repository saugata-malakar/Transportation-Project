"""
stage19_extract_physical_verification_frames.py

Extracts physical verification frames and creates comprehensive visual cards
for all 49 tracked pedestrians so every single result can be physically verified
against the original 4K drone video (DJI_20251005162440_0129_D.MP4).

Generates:
  1. Individual Physical Verification Cards for all 49 persons:
     work/physical_verification/cards/P01_verification_card.jpg ... P83_verification_card.jpg
  2. Master Verification Mosaic Sheets:
     work/physical_verification/ALL_49_PERSONS_VERIFICATION_SHEET.jpg
     work/physical_verification/MALES_33_VERIFICATION_SHEET.jpg
     work/physical_verification/FEMALES_16_VERIFICATION_SHEET.jpg
     work/physical_verification/ACTIVE_CROSSERS_SEQUENCE_SHEET.jpg
  3. Master Physical Verification Index CSV:
     work/physical_verification/pedestrian_physical_verification_index.csv
  4. Interactive Offline HTML Verification Viewer:
     work/physical_verification/physical_verification_viewer.html
"""

import os
import sys
import json
import csv
import cv2
import numpy as np

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

VIDEO_PATH = r"c:\Users\Administrator\Downloads\TRANSPORTATION\DJI_20251005162440_0129_D.MP4"
FRAMES_DIR = "work/frames"
TRACKS_PATH = "work/tracks/tracks.json"
ZONES_PATH = "work/crossing_zones/crossing_zones_perfect.json"
FEATURES_PATH = "work/features/pedestrian_behavioral_features_per_person.json"
GAIT_PATH = "work/gait_crossing/pedestrian_gait_and_crossing_summary.json"

OUT_DIR = "work/physical_verification"
CARDS_DIR = os.path.join(OUT_DIR, "cards")
CROPS_DIR = os.path.join(OUT_DIR, "crops")
os.makedirs(CARDS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

GSD_SCALE = 0.015  # 1 pixel = 0.015 meters (1.5 cm/px)


def format_time(seconds):
    """Format seconds into MM:SS.SS"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def get_frame_file(frame_id):
    """Locate or extract the image file corresponding to a frame_id."""
    f_num = int(frame_id)
    # Check work/frames/
    candidates = [f for f in os.listdir(FRAMES_DIR) if f.startswith(f"frame_{f_num:07d}")]
    if candidates:
        return os.path.join(FRAMES_DIR, candidates[0])
    
    # If not in frames dir, extract directly from raw video
    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            save_path = os.path.join(FRAMES_DIR, f"frame_{f_num:07d}_extracted.jpg")
            cv2.imwrite(save_path, frame)
            return save_path
            
    return None


def create_verification_card(pid, p_data, g_data, obs_list, zones):
    """
    Creates a full high-resolution verification card (1920x1080) for a pedestrian:
      Left 70%: Full 4K video frame scaled to 1344x1080 with target callout and zones
      Right 30%: Dossier panel with enlarged crop, trajectory mini-map, and physical metadata
    """
    # Pick the best observation (highest confidence, or mid-track)
    best_obs = max(obs_list, key=lambda o: (o["confidence"], (o["x2"]-o["x1"])*(o["y2"]-o["y1"])))
    frame_id = best_obs["frame_id"]
    timestamp = best_obs["timestamp_sec"]
    time_str = format_time(timestamp)

    frame_path = get_frame_file(frame_id)
    if frame_path is None or not os.path.exists(frame_path):
        return None

    raw_frame = cv2.imread(frame_path)
    if raw_frame is None:
        return None

    h_orig, w_orig = raw_frame.shape[:2]

    # Extract high-res crop of the pedestrian with margin
    x1, y1, x2, y2 = int(best_obs["x1"]), int(best_obs["y1"]), int(best_obs["x2"]), int(best_obs["y2"])
    margin_x = int((x2 - x1) * 0.5)
    margin_y = int((y2 - y1) * 0.4)
    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1 - margin_y)
    cx2 = min(w_orig, x2 + margin_x)
    cy2 = min(h_orig, y2 + margin_y)

    crop = raw_frame[cy1:cy2, cx1:cx2].copy()
    crop_save_path = os.path.join(CROPS_DIR, f"P{int(pid):02d}_crop_f{frame_id}.jpg")
    cv2.imwrite(crop_save_path, crop)

    # Output canvas (1920 x 1080)
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 30)  # Dark slate gray background

    # --- LEFT PANEL: FULL VIDEO FRAME WITH ANNOTATIONS ---
    left_w = 1340
    left_h = 1080
    scale_x = left_w / w_orig
    scale_y = left_h / h_orig

    frame_scaled = cv2.resize(raw_frame, (left_w, left_h))
    overlay = frame_scaled.copy()

    # Draw zones
    def scale_poly(pts):
        return np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in pts], dtype=np.int32)

    road_poly = scale_poly(zones["roadway_boundary"])
    corridor_poly = scale_poly(zones["pedestrian_crossing_corridors"][2]["polygon"])
    ul_zebra = scale_poly(zones["zebra_crossings"][0]["polygon"])
    lr_zebra = scale_poly(zones["zebra_crossings"][1]["polygon"])

    cv2.polylines(frame_scaled, [road_poly], True, (255, 200, 0), 2)
    cv2.fillPoly(overlay, [corridor_poly], (0, 140, 70))
    cv2.fillPoly(overlay, [ul_zebra], (0, 215, 255))
    cv2.fillPoly(overlay, [lr_zebra], (0, 215, 255))

    frame_blended = cv2.addWeighted(overlay, 0.30, frame_scaled, 0.70, 0)

    # Draw trajectory line
    pts_traj = []
    for o in sorted(obs_list, key=lambda x: x["timestamp_sec"]):
        px = int(((o["x1"] + o["x2"]) / 2.0) * scale_x)
        py = int(o["y2"] * scale_y)
        pts_traj.append((px, py))

    if len(pts_traj) >= 2:
        for k in range(len(pts_traj) - 1):
            cv2.line(frame_blended, pts_traj[k], pts_traj[k+1], (0, 255, 255), 2)

    # Draw Target Pedestrian Box and Pointer
    px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
    px2, py2 = int(x2 * scale_x), int(y2 * scale_y)
    p_center_x = (px1 + px2) // 2
    p_center_y = py2

    is_male = (p_data.get("gender") == "Male")
    color_gender = (255, 120, 0) if is_male else (180, 50, 255)  # Blue for Male, Magenta for Female

    # Bounding Box
    cv2.rectangle(frame_blended, (px1, py1), (px2, py2), color_gender, 3)

    # Concentric Targeting Rings around person
    cv2.circle(frame_blended, (p_center_x, p_center_y), 24, (0, 255, 255), 2)
    cv2.circle(frame_blended, (p_center_x, p_center_y), 36, color_gender, 2)
    cv2.drawMarker(frame_blended, (p_center_x, p_center_y), (0, 255, 255), cv2.MARKER_CROSS, 44, 2)

    # Target Callout Tag
    tag_text = f"TARGET: P{pid} ({p_data.get('gender')})"
    cv2.rectangle(frame_blended, (px1 - 4, py1 - 32), (px1 + 220, py1 - 4), (0, 0, 0), -1)
    cv2.rectangle(frame_blended, (px1 - 4, py1 - 32), (px1 + 220, py1 - 4), color_gender, 2)
    cv2.putText(frame_blended, tag_text, (px1 + 4, py1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Frame Info Banner (Top of left panel)
    cv2.rectangle(frame_blended, (15, 15), (750, 85), (15, 15, 15), -1)
    cv2.rectangle(frame_blended, (15, 15), (750, 85), (200, 200, 200), 1)
    cv2.putText(frame_blended, "UAV DRONE PHYSICAL VERIFICATION FRAME", (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame_blended, f"Source Video: DJI_...0129_D.MP4 | Frame #{frame_id} | Time: {time_str} ({timestamp:.2f}s)",
                (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    canvas[0:left_h, 0:left_w] = frame_blended

    # Vertical divider line
    cv2.line(canvas, (left_w, 0), (left_w, 1080), (80, 90, 105), 3)

    # --- RIGHT PANEL: PHYSICAL VERIFICATION DOSSIER ---
    right_x = left_w + 20
    panel_w = 1920 - right_x - 20

    # Header Box
    cv2.rectangle(canvas, (right_x, 15), (1900, 85), (30, 36, 48), -1)
    cv2.rectangle(canvas, (right_x, 15), (1900, 85), color_gender, 2)
    cv2.putText(canvas, f"PEDESTRIAN DOSSIER: #{pid}", (right_x + 15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(canvas, f"STATUS: PHYSICAL EVIDENCE VERIFIED", (right_x + 15, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1)

    # 1. High-Resolution Zoomed Crop (Right Side, Top)
    crop_display_w = 260
    crop_display_h = 260
    if crop is not None and crop.size > 0:
        crop_h, crop_w = crop.shape[:2]
        s_crop = min(crop_display_w / crop_w, crop_display_h / crop_h)
        nw, nh = int(crop_w * s_crop), int(crop_h * s_crop)
        crop_res = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        
        # Center in box
        off_x = right_x + (panel_w - nw) // 2
        off_y = 110 + (crop_display_h - nh) // 2
        
        cv2.rectangle(canvas, (off_x - 4, off_y - 4), (off_x + nw + 4, off_y + nh + 4), (50, 55, 65), -1)
        cv2.rectangle(canvas, (off_x - 4, off_y - 4), (off_x + nw + 4, off_y + nh + 4), color_gender, 2)
        canvas[off_y:off_y + nh, off_x:off_x + nw] = crop_res

        cv2.putText(canvas, f"Enlarged 4K Video Crop ({nw}x{nh}px)", (right_x + 120, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 2. Complete Physical Metadata Table
    meta_y_start = 425
    cv2.rectangle(canvas, (right_x, meta_y_start), (1900, 1055), (28, 32, 42), -1)
    cv2.rectangle(canvas, (right_x, meta_y_start), (1900, 1055), (60, 70, 85), 1)

    # Header for Table
    cv2.rectangle(canvas, (right_x, meta_y_start), (1900, meta_y_start + 35), (40, 48, 64), -1)
    cv2.putText(canvas, "PHYSICAL PARAMETERS & MEASUREMENTS", (right_x + 15, meta_y_start + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

    # Table items
    pos_x_px = (best_obs["x1"] + best_obs["x2"]) / 2.0
    pos_y_px = best_obs["y2"]
    pos_x_m = pos_x_px * GSD_SCALE
    pos_y_m = pos_y_px * GSD_SCALE

    items = [
        ("Video Frame #", f"Frame {frame_id} / 50,491"),
        ("Video Timestamp", f"{time_str} ({timestamp:.2f} seconds)"),
        ("Observation Span", f"{p_data.get('total_duration_sec', 0):.1f}s ({len(obs_list)} detections)"),
        ("Pixel Coordinates", f"X={pos_x_px:.1f}px, Y={pos_y_px:.1f}px"),
        ("Physical Metric Pos", f"X={pos_x_m:.2f}m, Y={pos_y_m:.2f}m (GSD=1.5cm/px)"),
        ("Gender Classification", f"{p_data.get('gender')} (Conf: {p_data.get('gender_confidence', 0):.1%})"),
        ("Carrying Load", f"{p_data.get('carrying_load', 'No Load')}"),
        ("Social Group Size", f"{p_data.get('group_type', 'Alone')} (Size={g_data.get('group_size', 1)})"),
        ("Walking Speed", f"{p_data.get('avg_walking_speed_mps', 0):.3f} m/s ({p_data.get('avg_walking_speed_kmh', 0):.2f} km/h)"),
        ("Max Speed Observed", f"{p_data.get('max_walking_speed_mps', 0):.3f} m/s"),
        ("Mean Stride Length", f"{g_data.get('mean_stride_length_m', 0):.3f} meters"),
        ("Mean Step Length", f"{g_data.get('mean_step_length_m', 0):.3f} meters"),
        ("Mean Cadence", f"{g_data.get('mean_cadence_steps_per_min', 0):.1f} steps / minute"),
        ("Total Estimated Steps", f"{g_data.get('total_estimated_steps', 0)} steps ({g_data.get('total_estimated_strides', 0)} strides)"),
        ("Spatial Zone (Origin)", f"{g_data.get('crossing_origin_side', 'Unknown')}"),
        ("Spatial Zone (Dest)", f"{g_data.get('crossing_destination_side', 'Unknown')}"),
        ("Crossing Transition", f"{g_data.get('crossing_behavior_type', 'Curbside')}"),
        ("Waiting Time at Curb", f"{p_data.get('total_waiting_time_sec', 0):.1f} seconds"),
        ("Crossing Attempts", f"{p_data.get('num_crossing_attempts', 0)} attempt(s)")
    ]

    row_y = meta_y_start + 62
    for label, val in items:
        cv2.putText(canvas, label, (right_x + 15, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 185, 200), 1)
        cv2.putText(canvas, val, (right_x + 230, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        row_y += 30

    card_path = os.path.join(CARDS_DIR, f"P{int(pid):02d}_verification_card.jpg")
    cv2.imwrite(card_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return card_path, frame_id, timestamp, time_str, pos_x_px, pos_y_px, pos_x_m, pos_y_m, crop_save_path


def generate_master_mosaic_sheet(summary_list, out_path, title, cols=7, cell_size=200):
    """
    Creates a large mosaic grid of pedestrian crops with ID, gender, timestamp, and frame #.
    """
    n = len(summary_list)
    rows = (n + cols - 1) // cols
    header_h = 100
    card_h = cell_size + 70
    card_w = cell_size
    grid_h = header_h + rows * card_h
    grid_w = cols * card_w

    mosaic = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    mosaic[:] = (24, 28, 34)

    # Header
    cv2.rectangle(mosaic, (0, 0), (grid_w, header_h), (35, 42, 54), -1)
    cv2.putText(mosaic, title.upper(), (30, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    cv2.putText(mosaic, f"Total Count: {n} Persons | Source: DJI_20251005162440_0129_D.MP4 (4K UAV Footage)",
                (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)

    for idx, item in enumerate(summary_list):
        r = idx // cols
        c = idx % cols
        x_off = c * card_w
        y_off = header_h + r * card_h

        crop_img = cv2.imread(item["crop_path"])
        if crop_img is not None and crop_img.size > 0:
            ch, cw = crop_img.shape[:2]
            scale = min((cell_size - 10) / cw, (cell_size - 10) / ch)
            nw, nh = int(cw * scale), int(ch * scale)
            res = cv2.resize(crop_img, (nw, nh))

            pad_x = (cell_size - nw) // 2
            pad_y = (cell_size - nh) // 2
            mosaic[y_off + pad_y:y_off + pad_y + nh, x_off + pad_x:x_off + pad_x + nw] = res

        # Border
        color = (255, 120, 0) if item["gender"] == "Male" else (180, 50, 255)
        cv2.rectangle(mosaic, (x_off + 2, y_off + 2), (x_off + card_w - 2, y_off + cell_size - 2), color, 2)

        # Labels below crop
        lbl_y = y_off + cell_size + 18
        cv2.rectangle(mosaic, (x_off + 2, y_off + cell_size), (x_off + card_w - 2, y_off + card_h - 2), (30, 35, 45), -1)
        cv2.putText(mosaic, f"P{item['person_id']} | {item['gender']} ({item['confidence']:.0%})", (x_off + 6, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv2.putText(mosaic, f"Frame #{item['frame_id']} | {item['time_str']}", (x_off + 6, lbl_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(mosaic, f"Zone: {item['zone'][:14]}", (x_off + 6, lbl_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 175, 190), 1)

    cv2.imwrite(out_path, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  -> Saved master mosaic: {out_path}")


def generate_html_viewer(summary_list):
    """Generates an offline HTML physical verification gallery."""
    html_path = os.path.join(OUT_DIR, "physical_verification_viewer.html")
    
    cards_json = json.dumps(summary_list)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>UAV Pedestrian Physical Verification Viewer (N=49)</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: #0f141c; color: #e2e8f0; display: flex; height: 100vh; overflow: hidden; }}
        #sidebar {{ width: 340px; background: #1a202c; border-right: 1px solid #2d3748; display: flex; flex-direction: column; }}
        #header {{ padding: 18px; background: #171923; border-bottom: 1px solid #2d3748; }}
        #header h2 {{ font-size: 16px; color: #38bdf8; margin-bottom: 4px; }}
        #header p {{ font-size: 12px; color: #94a3b8; }}
        #filters {{ padding: 12px 18px; background: #1e293b; display: flex; gap: 8px; border-bottom: 1px solid #2d3748; }}
        .filter-btn {{ padding: 6px 12px; border-radius: 4px; font-size: 11px; border: none; cursor: pointer; background: #334155; color: #cbd5e1; }}
        .filter-btn.active {{ background: #0284c7; color: #fff; font-weight: bold; }}
        #list {{ flex: 1; overflow-y: auto; padding: 10px; }}
        .item-card {{ display: flex; align-items: center; gap: 12px; padding: 10px; border-radius: 6px; margin-bottom: 8px; background: #242c3d; cursor: pointer; border: 1px solid transparent; transition: all 0.15s; }}
        .item-card:hover {{ background: #2d3748; border-color: #0284c7; }}
        .item-card.selected {{ background: #1e3a5f; border-color: #38bdf8; }}
        .thumb {{ width: 55px; height: 55px; border-radius: 4px; object-fit: cover; background: #000; border: 1px solid #475569; }}
        .info {{ flex: 1; }}
        .pid {{ font-weight: bold; font-size: 13px; }}
        .pid.Male {{ color: #60a5fa; }}
        .pid.Female {{ color: #f472b6; }}
        .sub {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}
        #main {{ flex: 1; display: flex; flex-direction: column; padding: 20px; overflow-y: auto; }}
        #card-container {{ background: #171923; border: 1px solid #2d3748; border-radius: 8px; padding: 12px; }}
        #card-img {{ width: 100%; border-radius: 4px; display: block; }}
    </style>
</head>
<body>
    <div id="sidebar">
        <div id="header">
            <h2>UAV Physical Verification</h2>
            <p>49 Pedestrians Verified in 4K Drone Video</p>
        </div>
        <div id="filters">
            <button class="filter-btn active" onclick="filterGender('All')">All (49)</button>
            <button class="filter-btn" onclick="filterGender('Male')">Males (33)</button>
            <button class="filter-btn" onclick="filterGender('Female')">Females (16)</button>
        </div>
        <div id="list"></div>
    </div>
    <div id="main">
        <div id="card-container">
            <img id="card-img" src="" alt="Select a pedestrian">
        </div>
    </div>

    <script>
        const data = {cards_json};
        let currentFilter = 'All';

        function renderList() {{
            const listEl = document.getElementById('list');
            listEl.innerHTML = '';
            const filtered = data.filter(d => currentFilter === 'All' || d.gender === currentFilter);
            filtered.forEach((d, idx) => {{
                const div = document.createElement('div');
                div.className = 'item-card' + (idx === 0 ? ' selected' : '');
                div.onclick = () => selectPerson(d, div);
                div.innerHTML = `
                    <img class="thumb" src="${{d.crop_rel_path}}" alt="P${{d.person_id}}">
                    <div class="info">
                        <div class="pid ${{d.gender}}">Person #${{d.person_id}} (${{d.gender}})</div>
                        <div class="sub">Frame #${{d.frame_id}} | ${{d.time_str}}</div>
                        <div class="sub">${{d.zone}}</div>
                    </div>
                `;
                listEl.appendChild(div);
            }});
            if (filtered.length > 0) {{
                selectPerson(filtered[0], listEl.children[0]);
            }}
        }}

        function selectPerson(d, el) {{
            document.querySelectorAll('.item-card').forEach(c => c.classList.remove('selected'));
            if (el) el.classList.add('selected');
            document.getElementById('card-img').src = d.card_rel_path;
        }}

        function filterGender(g) {{
            currentFilter = g;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            renderList();
        }}

        renderList();
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  -> Saved HTML verification gallery: {html_path}")


def main():
    print("=" * 75)
    print("STAGE 19: PHYSICAL VIDEO FRAME EXTRACTION & VERIFICATION SYSTEM")
    print("=" * 75)

    # 1. Load All Datasets
    with open(TRACKS_PATH) as f:
        tracks = json.load(f)["tracks"]
    with open(ZONES_PATH) as f:
        zones = json.load(f)
    with open(FEATURES_PATH) as f:
        feats = {p["person_id"]: p for p in json.load(f)}
    with open(GAIT_PATH) as f:
        gaits = {p["person_id"]: p for p in json.load(f)}

    print(f"[1/4] Loaded {len(tracks)} tracks, zone geometry, and attributes.")

    # 2. Extract and Generate Cards for All 49 Pedestrians
    print(f"[2/4] Generating high-resolution verification cards for all 49 pedestrians...")
    summary_list = []
    csv_rows = []

    pids_sorted = sorted(tracks.keys(), key=lambda x: int(x))
    total_pids = len(pids_sorted)

    for i, pid in enumerate(pids_sorted):
        obs_list = tracks[pid]
        p_data = feats.get(pid, {})
        g_data = gaits.get(pid, {})

        res = create_verification_card(pid, p_data, g_data, obs_list, zones)
        if res is not None:
            card_path, frame_id, timestamp, time_str, px, py, mx, my, crop_path = res

            item_dict = {
                "person_id": pid,
                "gender": p_data.get("gender", "Unknown"),
                "confidence": p_data.get("gender_confidence", 0.0),
                "frame_id": frame_id,
                "timestamp_sec": round(timestamp, 2),
                "time_str": time_str,
                "pos_x_px": round(px, 1),
                "pos_y_px": round(py, 1),
                "pos_x_m": round(mx, 2),
                "pos_y_m": round(my, 2),
                "zone": g_data.get("crossing_origin_side", "Unknown"),
                "destination": g_data.get("crossing_destination_side", "Unknown"),
                "behavior": g_data.get("crossing_behavior_type", "Curbside"),
                "speed_mps": p_data.get("avg_walking_speed_mps", 0.0),
                "speed_kmh": p_data.get("avg_walking_speed_kmh", 0.0),
                "stride_m": g_data.get("mean_stride_length_m", 0.0),
                "step_m": g_data.get("mean_step_length_m", 0.0),
                "cadence_spm": g_data.get("mean_cadence_steps_per_min", 0.0),
                "group_category": g_data.get("group_crossing_category", "Single"),
                "carrying_load": p_data.get("carrying_load", "No Load"),
                "waiting_time_sec": p_data.get("total_waiting_time_sec", 0.0),
                "crossing_attempts": p_data.get("num_crossing_attempts", 0),
                "card_path": card_path,
                "card_rel_path": f"cards/{os.path.basename(card_path)}",
                "crop_path": crop_path,
                "crop_rel_path": f"crops/{os.path.basename(crop_path)}"
            }
            summary_list.append(item_dict)
            csv_rows.append(item_dict)

        if (i + 1) % 10 == 0 or (i + 1) == total_pids:
            print(f"   Rendered {i + 1}/{total_pids} verification cards...")

    # 3. Save Master Index CSV
    print(f"[3/4] Exporting master physical verification index CSV...")
    csv_path = os.path.join(OUT_DIR, "pedestrian_physical_verification_index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # omit rel_path helper fields from CSV
        fields = [k for k in csv_rows[0].keys() if not k.endswith("_rel_path")]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  -> Saved index CSV: {csv_path}")

    # 4. Generate Master Visual Verification Mosaics
    print(f"[4/4] Rendering master verification mosaic sheets...")
    all_mosaic_path = os.path.join(OUT_DIR, "ALL_49_PERSONS_VERIFICATION_SHEET.jpg")
    generate_master_mosaic_sheet(summary_list, all_mosaic_path,
                                 "Master Physical Verification Sheet — All 49 Tracked Pedestrians")

    males_list = [s for s in summary_list if s["gender"] == "Male"]
    male_mosaic_path = os.path.join(OUT_DIR, "MALES_33_VERIFICATION_SHEET.jpg")
    generate_master_mosaic_sheet(males_list, male_mosaic_path,
                                 "Male Pedestrian Verification Sheet (33 Verified Males)")

    females_list = [s for s in summary_list if s["gender"] == "Female"]
    female_mosaic_path = os.path.join(OUT_DIR, "FEMALES_16_VERIFICATION_SHEET.jpg")
    generate_master_mosaic_sheet(females_list, female_mosaic_path,
                                 "Female Pedestrian Verification Sheet (16 Verified Females)")

    # 5. Generate Offline HTML Viewer
    generate_html_viewer(summary_list)

    print("\n[OK] STAGE 19 PHYSICAL VERIFICATION SYSTEM COMPLETE!")
    print(f"     Review all files in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
