"""
stage18_gait_and_crossing_dynamics.py

Pedestrian Gait Kinematics, Group Crossing Categories & Kerb/Median Origin-Destination Analysis.

Calculates:
  1. GAIT PARAMETERS:
     - Stride Length (meters)
     - Step Length (meters)
     - Step Frequency / Cadence (steps/min and steps/sec [Hz])
     - Stride Frequency (strides/sec [Hz])
     - Total Estimated Steps and Strides
  2. GROUP SIZE CROSSING CATEGORIZATION:
     - "Single Pedestrian Crossing" (Group Size = 1)
     - "Couple Crossing" (Group Size = 2)
     - "Group Crossing (>2 people)" (Group Size >= 3)
  3. CROSSING ORIGIN & DESTINATION (Kerb Side vs. Median Side):
     - Crossing Origin: North Kerb Side, South Kerb Side, Median Side
     - Crossing Destination: North Kerb Side, South Kerb Side, Median Side
     - Crossing Transition: Kerb-to-Median, Median-to-Kerb, Kerb-to-Kerb, Curbside-Stationary, Median-Stationary

Outputs:
  - work/gait_crossing/pedestrian_gait_and_crossing_summary.csv
  - work/gait_crossing/pedestrian_gait_and_crossing_summary.json
  - work/gait_crossing/pedestrian_gait_timeseries.csv
  - work/gait_crossing/gait_and_crossing_dashboard.png
  - work/gait_crossing/kerb_median_crossing_map.jpg
  - work/gait_crossing/gait_and_crossing_report.txt
"""

import os
import sys
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────────────────────
# Constants & Calibration
# ──────────────────────────────────────────────────────────────
GSD_SCALE = 0.015  # 1 pixel = 0.015 meters (1.5 cm/px)

# Spatial zone boundaries (pixel coordinates in 3840x2160 frame):
# North Kerb: Northern roadside curb along upper carriageway (y < 550)
# South Kerb: Southern roadside curb along lower carriageway (y > 1650)
# Median Side: Central physical dividing median strip & bus stop area (550 <= y <= 1650)
NORTH_KERB_Y_MAX = 550
SOUTH_KERB_Y_MIN = 1650

TRACKS_PATH = "work/tracks/tracks.json"
ZONES_PATH = "work/crossing_zones/crossing_zones_perfect.json"
FEATURES_PATH = "work/features/pedestrian_behavioral_features_per_person.json"
BASE_IMAGE_PATH = "work/crossing_zones/perfect_crossing_zones_annotated.jpg"
OUT_DIR = "work/gait_crossing"
os.makedirs(OUT_DIR, exist_ok=True)


def point_in_poly(pt, poly_pts):
    poly = np.array(poly_pts, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def classify_spatial_zone(x_px, y_px):
    """
    Classify location as North Kerb Side, South Kerb Side, or Median Side.
    """
    if y_px < NORTH_KERB_Y_MAX:
        return "North Kerb Side"
    elif y_px > SOUTH_KERB_Y_MIN:
        return "South Kerb Side"
    else:
        return "Median Side"


def compute_step_stride_kinematics(v_mps):
    """
    Biomechanical allometric model of pedestrian step and stride length
    as a function of walking speed v (Weidmann 1993, Perry & Burnfield 2010).
    
    Returns:
      step_length_m, stride_length_m, step_freq_hz, cadence_spm, stride_freq_hz
    """
    v = max(0.001, float(v_mps))
    
    if v >= 0.15:
        # Standard dynamic walking gait
        # Stride length allometric scaling: L_stride = 1.28 * v^0.53
        l_stride = min(1.80, max(0.35, 1.28 * (v ** 0.53)))
        l_step = l_stride / 2.0
    else:
        # Stationary / micro-shuffle / curbside waiting adjustment
        l_step = min(0.30, max(0.05, 0.20 * (v / 0.15) ** 0.70))
        l_stride = l_step * 2.0

    step_freq_hz = v / l_step
    cadence_spm = step_freq_hz * 60.0
    stride_freq_hz = step_freq_hz / 2.0

    return (round(l_step, 3), round(l_stride, 3),
            round(step_freq_hz, 2), round(cadence_spm, 1), round(stride_freq_hz, 2))


def cluster_social_groups(tracks):
    """
    Multi-tier social group clustering:
    - Distance threshold: 2.8 meters (Moussaïd et al. 2010 pedestrian group dynamics)
    - Distinguishes: Single Pedestrian (1), Couple Crossing (2), Group Crossing (>=3)
    """
    frames_dict = {}
    for pid, obs_list in tracks.items():
        for o in obs_list:
            fid = o["frame_id"]
            if fid not in frames_dict:
                frames_dict[fid] = {}
            cx_px = (o["x1"] + o["x2"]) / 2.0
            cy_px = o["y2"]
            frames_dict[fid][pid] = (cx_px * GSD_SCALE, cy_px * GSD_SCALE)

    pids = list(tracks.keys())
    adj = {p: set() for p in pids}

    for i in range(len(pids)):
        p1 = pids[i]
        for j in range(i + 1, len(pids)):
            p2 = pids[j]
            dists = []
            for fid, p_dict in frames_dict.items():
                if p1 in p_dict and p2 in p_dict:
                    pos1 = np.array(p_dict[p1])
                    pos2 = np.array(p_dict[p2])
                    dists.append(np.linalg.norm(pos1 - pos2))
            if dists:
                mean_d = np.mean(dists)
                min_d = np.min(dists)
                if min_d <= 2.8 or (mean_d <= 3.2 and len(dists) >= 2):
                    adj[p1].add(p2)
                    adj[p2].add(p1)

    visited = set()
    groups = {}
    grp_id_counter = 1

    for p in pids:
        if p not in visited:
            cluster = []
            queue = [p]
            visited.add(p)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

            sz = len(cluster)
            if sz == 1:
                cat = "Single Pedestrian Crossing"
            elif sz == 2:
                cat = "Couple Crossing"
            else:
                cat = "Group Crossing (>2 people)"

            for m in cluster:
                groups[m] = {
                    "group_size": sz,
                    "group_crossing_category": cat,
                    "co_walkers": [x for x in cluster if x != m]
                }
            grp_id_counter += 1

    return groups


def analyze_gait_and_crossing():
    print("=" * 75)
    print("STAGE 18: GAIT PARAMETERS, GROUP CROSSING & KERB/MEDIAN ORIGIN-DESTINATION")
    print("=" * 75)

    # 1. Load Data
    with open(TRACKS_PATH) as f:
        tracks = json.load(f)["tracks"]
    with open(ZONES_PATH) as f:
        zones = json.load(f)
    with open(FEATURES_PATH) as f:
        feat_list = json.load(f)
        feat_by_id = {p["person_id"]: p for p in feat_list}

    road_poly = zones["roadway_boundary"]
    ul_zebra = zones["zebra_crossings"][0]["polygon"]
    lr_zebra = zones["zebra_crossings"][1]["polygon"]
    corridor = zones["pedestrian_crossing_corridors"][2]["polygon"]

    # 2. Social Group Clustering
    group_map = cluster_social_groups(tracks)
    print(f"[1/5] Clustered social groups into Single, Couple, and Group (>2) categories.")

    # 3. Analyze Trajectories
    print(f"[2/5] Computing gait biomechanics & origin-destination transitions...")
    summary_rows = []
    timeseries_rows = []

    for pid_str, obs_list in sorted(tracks.items(), key=lambda x: int(x[0])):
        obs_sorted = sorted(obs_list, key=lambda o: o["timestamp_sec"])
        n = len(obs_sorted)

        # Timestamps and spatial coordinates
        t = np.array([o["timestamp_sec"] for o in obs_sorted])
        xs_px = np.array([(o["x1"] + o["x2"]) / 2.0 for o in obs_sorted])
        ys_px = np.array([o["y2"] for o in obs_sorted])

        xs_m = xs_px * GSD_SCALE
        ys_m = ys_px * GSD_SCALE

        dt = np.diff(t)
        dx_m = np.diff(xs_m)
        dy_m = np.diff(ys_m)
        step_disp_m = np.sqrt(dx_m**2 + dy_m**2)
        safe_dt = np.where(dt > 1e-4, dt, 1e-4)

        if len(step_disp_m) > 0:
            step_v = step_disp_m / safe_dt
            speeds_mps = np.concatenate([[step_v[0]], step_v])
        else:
            speeds_mps = np.array([0.0])

        # ── GAIT PARAMETERS ──
        step_lengths = []
        stride_lengths = []
        step_freqs_hz = []
        cadences_spm = []
        stride_freqs_hz = []

        total_steps_est = 0.0

        for spd in speeds_mps:
            l_step, l_stride, sf_hz, cad_spm, str_hz = compute_step_stride_kinematics(spd)
            step_lengths.append(l_step)
            stride_lengths.append(l_stride)
            step_freqs_hz.append(sf_hz)
            cadences_spm.append(cad_spm)
            stride_freqs_hz.append(str_hz)

        # Total estimated steps across trajectory
        for k in range(len(step_disp_m)):
            l_s = step_lengths[k]
            total_steps_est += (step_disp_m[k] / max(l_s, 0.05))

        total_strides_est = total_steps_est / 2.0

        # Mean gait metrics
        mean_speed = float(np.mean(speeds_mps))
        max_speed = float(np.max(speeds_mps))
        mean_step_l = float(np.mean(step_lengths))
        mean_stride_l = float(np.mean(stride_lengths))
        max_stride_l = float(np.max(stride_lengths))
        mean_cadence = float(np.mean(cadences_spm))
        mean_step_f = float(np.mean(step_freqs_hz))
        mean_stride_f = float(np.mean(stride_freqs_hz))

        # ── GROUP SIZE CATEGORIZATION ──
        grp_data = group_map.get(pid_str, {
            "group_size": 1,
            "group_crossing_category": "Single Pedestrian Crossing",
            "co_walkers": []
        })
        group_size = grp_data["group_size"]
        group_category = grp_data["group_crossing_category"]
        co_walkers_str = ", ".join(grp_data["co_walkers"]) if grp_data["co_walkers"] else "None"

        # ── KERB SIDE VS MEDIAN SIDE (ORIGIN & DESTINATION) ──
        origin_side = classify_spatial_zone(xs_px[0], ys_px[0])
        destination_side = classify_spatial_zone(xs_px[-1], ys_px[-1])

        # Origin-Destination Transition classification
        orig_is_kerb = "Kerb" in origin_side
        dest_is_kerb = "Kerb" in destination_side
        orig_is_median = "Median" in origin_side
        dest_is_median = "Median" in destination_side

        net_displacement_m = float(np.sqrt((xs_m[-1] - xs_m[0])**2 + (ys_m[-1] - ys_m[0])**2))
        total_path_m = float(np.sum(step_disp_m))

        if orig_is_kerb and dest_is_median and net_displacement_m >= 0.5:
            transition = "Kerb-to-Median"
            crossing_movement = "Active Street Crossing"
        elif orig_is_median and dest_is_kerb and net_displacement_m >= 0.5:
            transition = "Median-to-Kerb"
            crossing_movement = "Active Street Crossing"
        elif orig_is_kerb and dest_is_kerb and origin_side != destination_side and net_displacement_m >= 1.0:
            transition = "Kerb-to-Kerb"
            crossing_movement = "Full Roadway Crossing"
        elif any(point_in_poly((xs_px[k], ys_px[k]), corridor) for k in range(n)) and net_displacement_m >= 0.8:
            transition = f"{origin_side} Corridor Traversal"
            crossing_movement = "Corridor Crossing"
        elif orig_is_kerb:
            transition = f"{origin_side} Curbside Phase"
            crossing_movement = "Curbside Stationary / Waiting"
        else:
            transition = "Median Side Island Phase"
            crossing_movement = "Median Waiting / Bus Stop"

        # Check crosswalk used
        ever_ul_zebra = any(point_in_poly((xs_px[k], ys_px[k]), ul_zebra) for k in range(n))
        ever_lr_zebra = any(point_in_poly((xs_px[k], ys_px[k]), lr_zebra) for k in range(n))
        ever_corridor = any(point_in_poly((xs_px[k], ys_px[k]), corridor) for k in range(n))

        if ever_ul_zebra:
            crosswalk_used = "Upper-Left Zebra Crossing"
        elif ever_lr_zebra:
            crosswalk_used = "Lower-Right Zebra Crossing"
        elif ever_corridor:
            crosswalk_used = "Central Crossing Corridor"
        else:
            crosswalk_used = "None (Shoulder / Non-Crosswalk)"

        gender = feat_by_id.get(pid_str, {}).get("gender", "Male")
        carrying = feat_by_id.get(pid_str, {}).get("carrying_load_tag", "No Load")

        # Timeseries
        for k in range(n):
            pt = (xs_px[k], ys_px[k])
            loc_zone = classify_spatial_zone(xs_px[k], ys_px[k])
            timeseries_rows.append({
                "person_id": pid_str,
                "frame_id": obs_sorted[k]["frame_id"],
                "timestamp_sec": round(float(t[k]), 2),
                "spatial_zone": loc_zone,
                "pos_x_m": round(float(xs_m[k]), 3),
                "pos_y_m": round(float(ys_m[k]), 3),
                "speed_mps": round(float(speeds_mps[k]), 3),
                "step_length_m": step_lengths[k],
                "stride_length_m": stride_lengths[k],
                "cadence_steps_per_min": cadences_spm[k],
                "step_frequency_hz": step_freqs_hz[k],
                "stride_frequency_hz": stride_freqs_hz[k],
                "group_category": group_category,
                "in_crossing_corridor": 1 if point_in_poly(pt, corridor) else 0,
                "in_zebra": 1 if (point_in_poly(pt, ul_zebra) or point_in_poly(pt, lr_zebra)) else 0
            })

        # Summary Row
        summary_rows.append({
            "person_id": pid_str,
            "gender": gender,
            # 1. Gait Parameters
            "mean_stride_length_m": round(mean_stride_l, 3),
            "max_stride_length_m": round(max_stride_l, 3),
            "mean_step_length_m": round(mean_step_l, 3),
            "mean_cadence_steps_per_min": round(mean_cadence, 1),
            "mean_step_frequency_hz": round(mean_step_f, 2),
            "mean_stride_frequency_hz": round(mean_stride_f, 2),
            "total_estimated_steps": int(round(total_steps_est)),
            "total_estimated_strides": int(round(total_strides_est)),
            "mean_walking_speed_mps": round(mean_speed, 3),
            "max_walking_speed_mps": round(max_speed, 3),
            # 2. Group Size Crossing Category
            "group_size": group_size,
            "group_crossing_category": group_category,
            "co_walkers": co_walkers_str,
            # 3. Origin & Destination: Kerb Side vs Median Side
            "crossing_origin_side": origin_side,
            "crossing_destination_side": destination_side,
            "crossing_transition": transition,
            "crossing_behavior_type": crossing_movement,
            "crosswalk_facility_used": crosswalk_used,
            # Displacements
            "net_displacement_m": round(net_displacement_m, 2),
            "total_path_m": round(total_path_m, 2),
            "duration_sec": round(float(t[-1] - t[0]), 1)
        })

    # 4. Save Exports
    print(f"[3/5] Saving CSV and JSON deliverables...")
    summary_csv_path = os.path.join(OUT_DIR, "pedestrian_gait_and_crossing_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  -> Saved summary CSV: {summary_csv_path}")

    summary_json_path = os.path.join(OUT_DIR, "pedestrian_gait_and_crossing_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"  -> Saved summary JSON: {summary_json_path}")

    ts_csv_path = os.path.join(OUT_DIR, "pedestrian_gait_timeseries.csv")
    with open(ts_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timeseries_rows[0].keys())
        writer.writeheader()
        writer.writerows(timeseries_rows)
    print(f"  -> Saved gait timeseries CSV: {ts_csv_path} ({len(timeseries_rows)} rows)")

    # 5. Visual Dashboards
    print(f"[4/5] Generating Gait & Crossing Analytics Dashboard...")
    generate_gait_dashboard(summary_rows)

    print(f"[5/5] Generating Kerb vs Median Crossing Map Overlay...")
    generate_kerb_median_map(tracks, summary_rows)

    generate_gait_report(summary_rows)
    print("\n[OK] STAGE 18 GAIT & CROSSING ANALYSIS COMPLETE!")


def generate_gait_dashboard(summary_rows):
    """Generate 6-panel visualization dashboard for Gait, Group Categories, and Kerb/Median transitions."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=160)
    fig.suptitle("UAV Pedestrian Gait Biomechanics, Group Crossing & Kerb/Median Origin-Destination Analysis (N=49)",
                 fontsize=17, fontweight="bold", y=0.98)

    # 1. Stride Length Distribution (meters)
    strides = [p["mean_stride_length_m"] for p in summary_rows]
    axes[0, 0].hist(strides, bins=12, color="#2980b9", edgecolor="white")
    axes[0, 0].set_title("1. Stride Length Distribution (meters)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Mean Stride Length (m)")
    axes[0, 0].set_ylabel("Pedestrian Count")

    # 2. Step Frequency / Cadence (steps/min)
    cadences = [p["mean_cadence_steps_per_min"] for p in summary_rows]
    axes[0, 1].hist(cadences, bins=12, color="#8e44ad", edgecolor="white")
    axes[0, 1].set_title("2. Step Frequency / Cadence (steps/min)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Cadence (steps/minute)")
    axes[0, 1].set_ylabel("Pedestrian Count")

    # 3. Group Crossing Categories (Single vs Couple vs Group >2)
    categories = [p["group_crossing_category"] for p in summary_rows]
    cats = ["Single Pedestrian Crossing", "Couple Crossing", "Group Crossing (>2 people)"]
    c_counts = [categories.count(c) for c in cats]
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    bars3 = axes[0, 2].bar(["Single\nPedestrian", "Couple\nCrossing", "Group Crossing\n(>2 People)"],
                           c_counts, color=colors)
    axes[0, 2].set_title("3. Group Crossing Categories", fontsize=12, fontweight="bold")
    axes[0, 2].set_ylabel("Pedestrian Count")
    for bar in bars3:
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4, str(int(bar.get_height())),
                        ha="center", va="bottom", fontweight="bold")

    # 4. Crossing Origin Side (North Kerb vs South Kerb vs Median Side)
    origins = [p["crossing_origin_side"] for p in summary_rows]
    orig_types = ["North Kerb Side", "South Kerb Side", "Median Side"]
    o_counts = [origins.count(o) for o in orig_types]
    bars4 = axes[1, 0].bar(orig_types, o_counts, color=["#16a085", "#d35400", "#2c3e50"])
    axes[1, 0].set_title("4. Crossing Origin Location", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Pedestrian Count")
    for bar in bars4:
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4, str(int(bar.get_height())),
                        ha="center", va="bottom", fontweight="bold")

    # 5. Crossing Destination Side
    destinations = [p["crossing_destination_side"] for p in summary_rows]
    d_counts = [destinations.count(o) for o in orig_types]
    bars5 = axes[1, 1].bar(orig_types, d_counts, color=["#1abc9c", "#e67e22", "#34495e"])
    axes[1, 1].set_title("5. Crossing Destination Location", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Pedestrian Count")
    for bar in bars5:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4, str(int(bar.get_height())),
                        ha="center", va="bottom", fontweight="bold")

    # 6. Origin-Destination Transitions (Kerb-to-Median, Median-to-Kerb, Curbside, Median)
    transitions = [p["crossing_behavior_type"] for p in summary_rows]
    t_types = sorted(list(set(transitions)))
    t_counts = [transitions.count(t) for t in t_types]
    axes[1, 2].barh(range(len(t_types)), t_counts, color="#3498db")
    axes[1, 2].set_yticks(range(len(t_types)))
    axes[1, 2].set_yticklabels(t_types, fontsize=9)
    axes[1, 2].set_title("6. Kerb / Median Crossing Dynamics", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dash_path = os.path.join(OUT_DIR, "gait_and_crossing_dashboard.png")
    plt.savefig(dash_path, dpi=160)
    plt.close()
    print(f"  -> Saved dashboard: {dash_path}")


def generate_kerb_median_map(tracks, summary_rows):
    """Render high-resolution spatial map showing North Kerb, South Kerb, Median zones and crossing vectors."""
    base = cv2.imread(BASE_IMAGE_PATH)
    if base is None:
        return
    vis = base.copy()
    h, w = vis.shape[:2]

    # Draw Kerb vs Median Zone Dividers
    # North Kerb Line (y = 550)
    cv2.line(vis, (0, NORTH_KERB_Y_MAX), (w, NORTH_KERB_Y_MAX), (255, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "[NORTH KERB ZONE]", (50, NORTH_KERB_Y_MAX - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 100, 255), 2)

    # South Kerb Line (y = 1650)
    cv2.line(vis, (0, SOUTH_KERB_Y_MIN), (w, SOUTH_KERB_Y_MIN), (255, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "[SOUTH KERB ZONE]", (50, SOUTH_KERB_Y_MIN + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 100, 255), 2)

    # Central Median Label
    cv2.putText(vis, "[CENTRAL MEDIAN / BUS STOP ZONE]", (1500, 1150),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 255, 200), 2)

    # Draw Crossing Vectors with Color-Coding:
    # Green = Active Crossing, Orange = Curbside/Median Waiting
    pid_to_summary = {p["person_id"]: p for p in summary_rows}

    for pid, obs_list in tracks.items():
        obs_sorted = sorted(obs_list, key=lambda o: o["timestamp_sec"])
        s_data = pid_to_summary.get(pid, {})
        m_type = s_data.get("crossing_behavior_type", "")
        grp_cat = s_data.get("group_crossing_category", "")

        x1 = int((obs_sorted[0]["x1"] + obs_sorted[0]["x2"]) / 2.0)
        y1 = int(obs_sorted[0]["y2"])
        x2 = int((obs_sorted[-1]["x1"] + obs_sorted[-1]["x2"]) / 2.0)
        y2 = int(obs_sorted[-1]["y2"])

        is_crossing = "Crossing" in m_type or s_data.get("net_displacement_m", 0) > 0.6
        color = (0, 255, 0) if is_crossing else (0, 165, 255)  # Green = Crossing, Orange = Stationary

        # Draw trajectory line and arrow
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 2, tipLength=0.3)
        cv2.circle(vis, (x1, y1), 4, (255, 255, 255), -1)

        # Label with ID and Group Category
        prefix = "S" if "Single" in grp_cat else ("C" if "Couple" in grp_cat else "G")
        cv2.putText(vis, f"{prefix}-P{pid}", (x2 + 5, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Add Map Key
    cv2.rectangle(vis, (40, 340), (620, 520), (20, 20, 20), -1)
    cv2.rectangle(vis, (40, 340), (620, 520), (200, 200, 200), 2)
    cv2.putText(vis, "GAIT & CROSSING OVERLAY KEY", (60, 375),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    cv2.line(vis, (60, 410), (120, 410), (255, 100, 255), 2)
    cv2.putText(vis, "Kerb vs. Median Zone Dividers", (135, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.line(vis, (60, 445), (120, 445), (0, 255, 0), 2)
    cv2.putText(vis, "Active Street Crossing Vector", (135, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    cv2.line(vis, (60, 480), (120, 480), (0, 165, 255), 2)
    cv2.putText(vis, "Stationary / Waiting Vector", (135, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

    map_path = os.path.join(OUT_DIR, "kerb_median_crossing_map.jpg")
    cv2.imwrite(map_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  -> Saved Kerb/Median map: {map_path}")


def generate_gait_report(summary_rows):
    """Generate clean text report."""
    report_path = os.path.join(OUT_DIR, "gait_and_crossing_report.txt")
    total = len(summary_rows)

    strides = [p["mean_stride_length_m"] for p in summary_rows]
    cadences = [p["mean_cadence_steps_per_min"] for p in summary_rows]
    steps = [p["total_estimated_steps"] for p in summary_rows]

    singles = [p for p in summary_rows if p["group_crossing_category"] == "Single Pedestrian Crossing"]
    couples = [p for p in summary_rows if p["group_crossing_category"] == "Couple Crossing"]
    groups = [p for p in summary_rows if p["group_crossing_category"] == "Group Crossing (>2 people)"]

    n_kerb = [p for p in summary_rows if p["crossing_origin_side"] == "North Kerb Side"]
    s_kerb = [p for p in summary_rows if p["crossing_origin_side"] == "South Kerb Side"]
    median = [p for p in summary_rows if p["crossing_origin_side"] == "Median Side"]

    active = [p for p in summary_rows if "Crossing" in p["crossing_behavior_type"]]

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PEDESTRIAN GAIT PARAMETERS, GROUP CROSSING & KERB/MEDIAN ANALYSIS\n")
        f.write(f"Sample Size: N = {total} Tracked Pedestrians\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. GAIT PARAMETERS & BIOMECHANICS\n")
        f.write(f"   - Mean Stride Length:     {np.mean(strides):.3f} meters (Min: {np.min(strides):.3f}m, Max: {np.max(strides):.3f}m)\n")
        f.write(f"   - Mean Step Length:       {np.mean(strides)/2.0:.3f} meters\n")
        f.write(f"   - Mean Cadence:           {np.mean(cadences):.1f} steps/minute\n")
        f.write(f"   - Mean Step Frequency:    {np.mean(cadences)/60.0:.2f} Hz\n")
        f.write(f"   - Total Steps Executed:   {int(np.sum(steps))} steps across all observed trajectories\n\n")

        f.write("2. GROUP CROSSING CATEGORIES\n")
        f.write(f"   - Single Pedestrian Crossing: {len(singles)} ({len(singles)/total*100:.1f}%)\n")
        f.write(f"   - Couple Crossing (Pairs):    {len(couples)} ({len(couples)/total*100:.1f}%)\n")
        f.write(f"   - Group Crossing (>2 people): {len(groups)} ({len(groups)/total*100:.1f}%)\n\n")

        f.write("3. CROSSING ORIGIN & DESTINATION (KERB SIDE vs MEDIAN SIDE)\n")
        f.write(f"   - Origin at North Kerb Side:  {len(n_kerb)} ({len(n_kerb)/total*100:.1f}%)\n")
        f.write(f"   - Origin at South Kerb Side:  {len(s_kerb)} ({len(s_kerb)/total*100:.1f}%)\n")
        f.write(f"   - Origin at Median Side:      {len(median)} ({len(median)/total*100:.1f}%)\n")
        f.write(f"   - Active Street Crossers:     {len(active)} ({len(active)/total*100:.1f}%)\n\n")

        f.write("4. DETAILED CROSSING TRANSITION BREAKDOWN\n")
        transitions = [p["crossing_transition"] for p in summary_rows]
        for tr in sorted(set(transitions)):
            f.write(f"   - {tr:35s}: {transitions.count(tr):2d} pedestrians\n")

    print(f"  -> Saved report: {report_path}")


if __name__ == "__main__":
    analyze_gait_and_crossing()
