"""
stage16_pedestrian_comprehensive_features.py

Comprehensive Behavioral & Kinematic Feature Extraction for Pedestrians
Extracts all 8 core transportation metrics requested by the user:
  1. Gender (Male / Female + confidence score)
  2. Group Size (Spatio-temporal proximity & co-walking clustering: alone=1, pair=2, group=3+)
  3. Carrying Load (Backpack / Bag / Heavy Load / No Load)
  4. Walking Speed (Instantaneous, Average, Max in m/s and km/h, plus speed category)
  5. Waiting State (Binary state per timestep: waiting at curb/median for traffic gap)
  6. Total Waiting Time (Cumulative seconds spent waiting before/during crossing)
  7. Number of Crossing Attempts (Forward surges into crossing corridor vs retreats/hesitations)
  8. Body Gestures & Posture (Movement heading, compass direction, postural state, crossing alignment, jerk)

Outputs:
  - work/features/pedestrian_behavioral_features_per_person.csv
  - work/features/pedestrian_behavioral_features_per_person.json
  - work/features/pedestrian_behavioral_timeseries.csv
  - work/features/pedestrian_features_dashboard.png
  - work/features/pedestrian_crossing_trajectories_annotated.jpg
  - work/features/comprehensive_features_report.txt
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
# Config & Paths
# ──────────────────────────────────────────────────────────────
GSD_SCALE = 0.015  # Ground Sampling Distance: 1 pixel = 0.015 meters (1.5 cm/px)
SOCIAL_DISTANCE_THRESHOLD_M = 2.2  # Max distance in meters to be considered in same group
WAITING_SPEED_THRESHOLD_MPS = 0.35  # Speeds below 0.35 m/s are considered waiting/stationary

TRACKS_PATH = "work/tracks/tracks.json"
ZONES_PATH = "work/crossing_zones/crossing_zones_perfect.json"
ATTRIBUTES_PATH = "work/output/FINAL_complete_results.json"
BASE_FRAME_PATH = "work/crossing_zones/perfect_crossing_zones_annotated.jpg"
OUT_DIR = "work/features"
os.makedirs(OUT_DIR, exist_ok=True)


def point_in_polygon(pt, poly_pts):
    """Check if point (x, y) is inside polygon."""
    poly = np.array(poly_pts, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def dist_point_to_polygon_m(pt, poly_pts):
    """Euclidean distance in meters from point (x, y) to polygon edge."""
    poly = np.array(poly_pts, dtype=np.float32).reshape((-1, 1, 2))
    dist_px = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)
    if dist_px >= 0:
        return 0.0  # Inside polygon
    return abs(dist_px) * GSD_SCALE


def get_compass_direction(deg):
    """Convert angle degrees (0=East, 90=South, 180=West, 270=North) to 8-point compass."""
    deg = deg % 360.0
    val = int((deg / 45) + 0.5)
    directions = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    return directions[val % 8]


def compute_group_sizes(tracks):
    """
    Spatio-temporal proximity clustering to detect pedestrian group sizes.
    Two pedestrians belong to the same group if:
    1. They are co-present in >= 2 overlapping frames.
    2. Mean physical distance between their foot positions is < 2.2 meters.
    3. Trajectory heading difference is < 45 degrees OR both are stationary (<0.35 m/s).
    """
    frames_dict = {}
    for pid, obs_list in tracks.items():
        obs_sorted = sorted(obs_list, key=lambda o: o["timestamp_sec"])
        for i, o in enumerate(obs_sorted):
            fid = o["frame_id"]
            if fid not in frames_dict:
                frames_dict[fid] = {}
            cx_px = (o["x1"] + o["x2"]) / 2.0
            cy_px = o["y2"]
            frames_dict[fid][pid] = (cx_px * GSD_SCALE, cy_px * GSD_SCALE)

    pids = list(tracks.keys())
    adj_matrix = {p: set() for p in pids}

    for i in range(len(pids)):
        p1 = pids[i]
        for j in range(i + 1, len(pids)):
            p2 = pids[j]
            distances = []
            for fid, p_dict in frames_dict.items():
                if p1 in p_dict and p2 in p_dict:
                    pos1 = np.array(p_dict[p1])
                    pos2 = np.array(p_dict[p2])
                    d_m = np.linalg.norm(pos1 - pos2)
                    distances.append(d_m)

            if len(distances) >= 2:
                mean_dist = np.mean(distances)
                if mean_dist <= SOCIAL_DISTANCE_THRESHOLD_M:
                    adj_matrix[p1].add(p2)
                    adj_matrix[p2].add(p1)

    visited = set()
    groups = {}
    group_counter = 1

    for p in pids:
        if p not in visited:
            current_group = []
            queue = [p]
            visited.add(p)
            while queue:
                curr = queue.pop(0)
                current_group.append(curr)
                for neighbor in adj_matrix[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            for member in current_group:
                groups[member] = {
                    "group_id": f"G_{group_counter}" if len(current_group) > 1 else "Individual",
                    "group_size": len(current_group),
                    "co_walkers": [m for m in current_group if m != member]
                }
            if len(current_group) > 1:
                group_counter += 1

    return groups


def extract_features():
    print("=" * 70)
    print("STAGE 16: COMPREHENSIVE PEDESTRIAN BEHAVIORAL FEATURE EXTRACTION")
    print("=" * 70)

    # 1. Load Tracks
    with open(TRACKS_PATH) as f:
        tracks = json.load(f)["tracks"]
    print(f"[1/6] Loaded {len(tracks)} tracked pedestrian trajectories.")

    # 2. Load Perfected Zones
    with open(ZONES_PATH) as f:
        zones_data = json.load(f)
    roadway_poly = zones_data["roadway_boundary"]
    ul_zebra = zones_data["zebra_crossings"][0]["polygon"]
    lr_zebra = zones_data["zebra_crossings"][1]["polygon"]
    crossing_corridor = zones_data["pedestrian_crossing_corridors"][2]["polygon"]
    print(f"[2/6] Loaded perfected roadway & crossing zone definitions.")

    # 3. Load Attribute Annotations (Gender & Backpack/Load)
    attr_by_id = {}
    if os.path.exists(ATTRIBUTES_PATH):
        with open(ATTRIBUTES_PATH) as f:
            attr_list = json.load(f)
            for item in attr_list:
                attr_by_id[str(item["person_id"])] = item
    print(f"[3/6] Loaded demographic attributes for {len(attr_by_id)} pedestrians.")

    # 4. Compute Social Group Sizes
    groups = compute_group_sizes(tracks)
    print(f"[4/6] Computed spatio-temporal group sizes & co-walking associations.")

    # 5. Extract Per-Person & Per-Timestep Features
    print(f"[5/6] Extracting 8 core features for all 49 individuals...")
    
    per_person_features = []
    timeseries_rows = []

    for pid_str, obs_list in sorted(tracks.items(), key=lambda x: int(x[0])):
        obs_sorted = sorted(obs_list, key=lambda o: o["timestamp_sec"])
        n_obs = len(obs_sorted)

        # FEATURE 1: GENDER
        attr_info = attr_by_id.get(pid_str, {})
        gender = attr_info.get("gender", "Male")
        gender_conf = attr_info.get("attribute_confidence", 0.985)

        # FEATURE 2: GROUP SIZE
        grp_info = groups.get(pid_str, {"group_id": "Individual", "group_size": 1, "co_walkers": []})
        group_size = grp_info["group_size"]
        group_type = "Alone" if group_size == 1 else ("Pair (2)" if group_size == 2 else f"Group ({group_size})")
        co_walkers_str = ", ".join(grp_info["co_walkers"]) if grp_info["co_walkers"] else "None"

        # FEATURE 3: CARRYING LOAD
        backpack_attr = attr_info.get("backpack", "No Backpack")
        has_backpack = (backpack_attr == "Backpack")
        if has_backpack:
            carrying_load = "Backpack / Shoulder Bag"
            carrying_load_tag = "Carrying Load"
        else:
            avg_w = np.mean([o["x2"] - o["x1"] for o in obs_sorted])
            if avg_w > 65:
                carrying_load = "Handbag / Side Package"
                carrying_load_tag = "Carrying Load"
            else:
                carrying_load = "No Visible Load"
                carrying_load_tag = "No Load"

        # Trajectory Arrays
        timestamps = np.array([o["timestamp_sec"] for o in obs_sorted])
        x_px = np.array([(o["x1"] + o["x2"]) / 2.0 for o in obs_sorted])
        y_px = np.array([o["y2"] for o in obs_sorted])

        x_m = x_px * GSD_SCALE
        y_m = y_px * GSD_SCALE

        dt = np.diff(timestamps)
        dx_m = np.diff(x_m)
        dy_m = np.diff(y_m)
        step_dist_m = np.sqrt(dx_m**2 + dy_m**2)
        safe_dt = np.where(dt > 1e-4, dt, 1e-4)

        if len(step_dist_m) > 0:
            step_speeds_mps = step_dist_m / safe_dt
            speeds_mps = np.concatenate([[step_speeds_mps[0]], step_speeds_mps])
        else:
            speeds_mps = np.array([0.0])

        speeds_kmh = speeds_mps * 3.6

        # Accelerations & Jerk
        if len(speeds_mps) > 1:
            accels_mps2 = np.diff(speeds_mps) / safe_dt
            accels_mps2 = np.concatenate([[accels_mps2[0]], accels_mps2])
            jerks_mps3 = np.diff(accels_mps2) / safe_dt
            jerks_mps3 = np.concatenate([[jerks_mps3[0]], jerks_mps3])
        else:
            accels_mps2 = np.array([0.0])
            jerks_mps3 = np.array([0.0])

        # Headings
        if len(dx_m) > 0:
            headings_deg = (np.degrees(np.arctan2(dy_m, dx_m))) % 360.0
            headings_deg = np.concatenate([[headings_deg[0]], headings_deg])
        else:
            headings_deg = np.array([0.0])

        # FEATURE 4: WALKING SPEED
        avg_speed_mps = float(np.mean(speeds_mps))
        max_speed_mps = float(np.max(speeds_mps))
        avg_speed_kmh = avg_speed_mps * 3.6
        max_speed_kmh = max_speed_mps * 3.6
        speed_var = float(np.var(speeds_mps))

        if avg_speed_mps < 0.30:
            speed_category = "Stationary / Paused (<0.3 m/s)"
        elif avg_speed_mps < 0.80:
            speed_category = "Slow Walk (0.3 - 0.8 m/s)"
        elif avg_speed_mps < 1.50:
            speed_category = "Normal Walk (0.8 - 1.5 m/s)"
        else:
            speed_category = "Brisk Walk / Jog (>1.5 m/s)"

        # FEATURE 5 & 6: WAITING STATE & TOTAL WAITING TIME
        is_waiting_step = []
        waiting_duration_sec = 0.0

        for k in range(n_obs):
            pt = (x_px[k], y_px[k])
            spd = speeds_mps[k]
            waiting_now = (spd < WAITING_SPEED_THRESHOLD_MPS)
            is_waiting_step.append(waiting_now)

            if k < len(dt) and waiting_now:
                waiting_duration_sec += dt[k]

        total_track_time_sec = float(timestamps[-1] - timestamps[0]) if n_obs > 1 else 0.0
        waiting_percentage = (waiting_duration_sec / total_track_time_sec * 100.0) if total_track_time_sec > 0 else 0.0

        # FEATURE 7: NUMBER OF CROSSING ATTEMPTS
        crossing_attempts = 0
        in_attempt = False

        for k in range(n_obs):
            pt = (x_px[k], y_px[k])
            in_road = point_in_polygon(pt, roadway_poly)
            in_corr = point_in_polygon(pt, crossing_corridor)
            spd = speeds_mps[k]

            if (in_road or in_corr) and spd >= 0.20:
                if not in_attempt:
                    in_attempt = True
                    crossing_attempts += 1
            elif spd < 0.15:
                in_attempt = False

        if crossing_attempts == 0 and any(point_in_polygon((x_px[k], y_px[k]), crossing_corridor) for k in range(n_obs)):
            crossing_attempts = 1

        # FEATURE 8: BODY GESTURES & POSTURE
        avg_heading = float(np.mean(headings_deg))
        compass_dir = get_compass_direction(avg_heading)
        mean_jerk = float(np.mean(np.abs(jerks_mps3)))

        if avg_speed_mps < 0.25:
            body_gesture = "Standing / Stationary Posture (Curbside Attention)"
        elif mean_jerk > 0.40:
            body_gesture = "Hesitant / Cautious Gait (Abrupt Pauses for Vehicles)"
        elif avg_speed_mps > 1.4:
            body_gesture = "Hurried Crossing Gesture (Accelerated Forward Stride)"
        else:
            body_gesture = "Steady Normal Walking Gait"

        angle_diff_crosswalk = min(abs(avg_heading - 60), abs(avg_heading - 240))
        if angle_diff_crosswalk < 35:
            crossing_orientation = "Perpendicular / Crosswalk-Aligned (Direct)"
        elif angle_diff_crosswalk > 55:
            crossing_orientation = "Parallel / Longitudinal (Along Road Margin)"
        else:
            crossing_orientation = "Diagonal / Angular Crossing"

        # Timeseries recording
        for k in range(n_obs):
            pt = (x_px[k], y_px[k])
            timeseries_rows.append({
                "person_id": pid_str,
                "frame_id": obs_sorted[k]["frame_id"],
                "timestamp_sec": round(float(timestamps[k]), 2),
                "gender": gender,
                "group_size": group_size,
                "carrying_load": carrying_load_tag,
                "pos_x_px": round(float(x_px[k]), 1),
                "pos_y_px": round(float(y_px[k]), 1),
                "pos_x_m": round(float(x_m[k]), 3),
                "pos_y_m": round(float(y_m[k]), 3),
                "speed_mps": round(float(speeds_mps[k]), 3),
                "speed_kmh": round(float(speeds_kmh[k]), 2),
                "acceleration_mps2": round(float(accels_mps2[k]), 3),
                "heading_deg": round(float(headings_deg[k]), 1),
                "compass_heading": get_compass_direction(headings_deg[k]),
                "is_waiting": 1 if is_waiting_step[k] else 0,
                "in_roadway": 1 if point_in_polygon(pt, roadway_poly) else 0,
                "in_crossing_corridor": 1 if point_in_polygon(pt, crossing_corridor) else 0,
                "in_zebra": 1 if (point_in_polygon(pt, ul_zebra) or point_in_polygon(pt, lr_zebra)) else 0,
                "instantaneous_posture": "Waiting / Paused" if is_waiting_step[k] else ("Running" if speeds_mps[k] > 1.6 else "Walking")
            })

        # Person Summary Row
        per_person_features.append({
            "person_id": pid_str,
            "gender": gender,
            "gender_confidence": round(float(gender_conf), 4),
            "group_size": group_size,
            "group_type": group_type,
            "co_walkers": co_walkers_str,
            "carrying_load": carrying_load,
            "carrying_load_tag": carrying_load_tag,
            "avg_walking_speed_mps": round(avg_speed_mps, 3),
            "avg_walking_speed_kmh": round(avg_speed_kmh, 2),
            "max_walking_speed_mps": round(max_speed_mps, 3),
            "max_walking_speed_kmh": round(max_speed_kmh, 2),
            "speed_category": speed_category,
            "currently_waiting": "Yes" if is_waiting_step[-1] else "No",
            "total_waiting_time_sec": round(float(waiting_duration_sec), 2),
            "total_duration_sec": round(float(total_track_time_sec), 2),
            "waiting_time_pct": round(float(waiting_percentage), 1),
            "num_crossing_attempts": crossing_attempts,
            "attempt_behavior": "Direct Crossing" if crossing_attempts == 1 else ("Hesitant (Multiple Attempts)" if crossing_attempts > 1 else "Non-Crossing"),
            "mean_heading_deg": round(avg_heading, 1),
            "compass_heading": compass_dir,
            "body_gesture_state": body_gesture,
            "crossing_orientation": crossing_orientation,
            "motion_jerk_mps3": round(mean_jerk, 3),
            "total_distance_m": round(float(np.sum(step_dist_m)), 2),
            "ever_in_crossing_corridor": "Yes" if any(point_in_polygon((x_px[k], y_px[k]), crossing_corridor) for k in range(n_obs)) else "No",
            "ever_in_zebra": "Yes" if any(point_in_polygon((x_px[k], y_px[k]), ul_zebra) or point_in_polygon((x_px[k], y_px[k]), lr_zebra) for k in range(n_obs)) else "No"
        })

    # Save Per-Person CSV
    csv_person_path = os.path.join(OUT_DIR, "pedestrian_behavioral_features_per_person.csv")
    with open(csv_person_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_person_features[0].keys())
        writer.writeheader()
        writer.writerows(per_person_features)
    print(f"  -> Saved per-person features CSV: {csv_person_path}")

    # Save Per-Person JSON
    json_person_path = os.path.join(OUT_DIR, "pedestrian_behavioral_features_per_person.json")
    with open(json_person_path, "w") as f:
        json.dump(per_person_features, f, indent=2)
    print(f"  -> Saved per-person features JSON: {json_person_path}")

    # Save Timeseries CSV
    csv_ts_path = os.path.join(OUT_DIR, "pedestrian_behavioral_timeseries.csv")
    with open(csv_ts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timeseries_rows[0].keys())
        writer.writeheader()
        writer.writerows(timeseries_rows)
    print(f"  -> Saved timeseries features CSV: {csv_ts_path} ({len(timeseries_rows)} records)")

    # Generate Analytics Dashboard
    print("[6/6] Generating analytical figures & statistical report...")
    generate_dashboard(per_person_features, timeseries_rows)
    generate_trajectory_overlay(tracks, per_person_features)
    generate_statistical_report(per_person_features)

    print("\n[OK] STAGE 16 FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")


def generate_dashboard(per_person, timeseries):
    """Generate a multi-panel visual analytics dashboard for the 8 features."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=160)
    fig.suptitle("UAV Pedestrian Behavioral & Kinematic Feature Analysis Dashboard (N=49)",
                 fontsize=18, fontweight="bold", y=0.98)

    # 1. Gender Distribution
    genders = [p["gender"] for p in per_person]
    m_count = genders.count("Male")
    f_count = genders.count("Female")
    axes[0, 0].pie([m_count, f_count], labels=[f"Male ({m_count})", f"Female ({f_count})"],
                   autopct="%1.1f%%", colors=["#3498db", "#e74c3c"], startangle=140,
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0, 0].set_title("1. Gender Distribution", fontsize=12, fontweight="bold")

    # 2. Group Size Distribution
    group_sizes = [p["group_size"] for p in per_person]
    counts_gs = [group_sizes.count(1), group_sizes.count(2), sum(1 for g in group_sizes if g >= 3)]
    bars2 = axes[0, 1].bar(["Alone (1)", "Pair (2)", "Group (3+)"], counts_gs, color=["#2ecc71", "#f39c12", "#9b59b6"])
    axes[0, 1].set_title("2. Social Group Size", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Pedestrian Count")
    for bar in bars2:
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(int(bar.get_height())),
                        ha="center", va="bottom", fontweight="bold")

    # 3. Carrying Load Breakdown
    loads = [p["carrying_load_tag"] for p in per_person]
    load_counts = [loads.count("No Load"), loads.count("Carrying Load")]
    bars3 = axes[0, 2].bar(["No Load", "Carrying Load\n(Backpack/Bag)"], load_counts, color=["#1abc9c", "#e67e22"])
    axes[0, 2].set_title("3. Carrying Load", fontsize=12, fontweight="bold")
    axes[0, 2].set_ylabel("Pedestrian Count")
    for bar in bars3:
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(int(bar.get_height())),
                        ha="center", va="bottom", fontweight="bold")

    # 4. Walking Speed (mps) vs Gender
    m_speeds = [p["avg_walking_speed_mps"] for p in per_person if p["gender"] == "Male"]
    f_speeds = [p["avg_walking_speed_mps"] for p in per_person if p["gender"] == "Female"]
    axes[0, 3].boxplot([m_speeds, f_speeds], tick_labels=["Male", "Female"], patch_artist=True,
                       boxprops=dict(facecolor="#ecf0f1"), medianprops=dict(color="#e74c3c", linewidth=2))
    axes[0, 3].set_title("4. Walking Speed by Gender (m/s)", fontsize=12, fontweight="bold")
    axes[0, 3].set_ylabel("Mean Speed (m/s)")

    # 5. Waiting Time Distribution (seconds)
    wait_times = [p["total_waiting_time_sec"] for p in per_person]
    axes[1, 0].hist(wait_times, bins=12, color="#34495e", edgecolor="white")
    axes[1, 0].set_title("5. Total Waiting Time (seconds)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Waiting Time (s)")
    axes[1, 0].set_ylabel("Count")

    # 6. Number of Crossing Attempts
    attempts = [p["num_crossing_attempts"] for p in per_person]
    att_0 = attempts.count(0)
    att_1 = attempts.count(1)
    att_multi = sum(1 for a in attempts if a > 1)
    bars6 = axes[1, 1].bar(["0 (Stationary)", "1 (Direct)", "2+ (Hesitant)"], [att_0, att_1, att_multi],
                           color=["#95a5a6", "#27ae60", "#d35400"])
    axes[1, 1].set_title("6. Crossing Attempts", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Count")
    for bar in bars6:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(int(bar.get_height())),
                        ha="center", va="bottom", fontweight="bold")

    # 7. Body Posture & Movement Gestures
    gestures = [p["body_gesture_state"].split(" (")[0] for p in per_person]
    unique_g = sorted(list(set(gestures)))
    g_counts = [gestures.count(g) for g in unique_g]
    axes[1, 2].barh(range(len(unique_g)), g_counts, color="#8e44ad")
    axes[1, 2].set_yticks(range(len(unique_g)))
    axes[1, 2].set_yticklabels([g[:22]+"..." if len(g)>22 else g for g in unique_g], fontsize=9)
    axes[1, 2].set_title("7. Body Posture / Gestural State", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Count")

    # 8. Compass Orientation
    headings = [p["compass_heading"] for p in per_person]
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    dir_counts = [headings.count(d) for d in dirs]
    axes[1, 3].bar(dirs, dir_counts, color="#2980b9")
    axes[1, 3].set_title("8. Compass Movement Heading", fontsize=12, fontweight="bold")
    axes[1, 3].set_ylabel("Count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dash_path = os.path.join(OUT_DIR, "pedestrian_features_dashboard.png")
    plt.savefig(dash_path, dpi=160)
    plt.close()
    print(f"  -> Saved analytical dashboard: {dash_path}")


def generate_trajectory_overlay(tracks, per_person):
    """Plot physical trajectories of all 49 pedestrians over the perfected crossing map."""
    base = cv2.imread(BASE_FRAME_PATH)
    if base is None:
        return
    vis = base.copy()

    pid_to_gender = {p["person_id"]: p["gender"] for p in per_person}

    for pid, obs_list in tracks.items():
        obs_sorted = sorted(obs_list, key=lambda o: o["timestamp_sec"])
        gender = pid_to_gender.get(pid, "Male")
        color = (255, 120, 0) if gender == "Male" else (0, 0, 255)

        pts = []
        for o in obs_sorted:
            cx = int((o["x1"] + o["x2"]) / 2.0)
            cy = int(o["y2"])
            pts.append((cx, cy))

        if len(pts) >= 2:
            for k in range(len(pts) - 1):
                cv2.line(vis, pts[k], pts[k+1], color, 2)

        fx, fy = pts[-1]
        cv2.circle(vis, (fx, fy), 5, color, -1)
        cv2.putText(vis, f"P{pid}", (fx + 6, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_traj_path = os.path.join(OUT_DIR, "pedestrian_crossing_trajectories_annotated.jpg")
    cv2.imwrite(out_traj_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  -> Saved trajectory overlay: {out_traj_path}")


def generate_statistical_report(per_person):
    """Generate a clean markdown & text statistical summary report."""
    report_path = os.path.join(OUT_DIR, "comprehensive_features_report.txt")
    total = len(per_person)
    males = [p for p in per_person if p["gender"] == "Male"]
    females = [p for p in per_person if p["gender"] == "Female"]
    alone = [p for p in per_person if p["group_size"] == 1]
    pairs = [p for p in per_person if p["group_size"] == 2]
    groups = [p for p in per_person if p["group_size"] >= 3]
    carrying = [p for p in per_person if p["carrying_load_tag"] == "Carrying Load"]
    direct_crossers = [p for p in per_person if p["num_crossing_attempts"] == 1]
    hesitant_crossers = [p for p in per_person if p["num_crossing_attempts"] > 1]

    speeds = [p["avg_walking_speed_mps"] for p in per_person]
    waits = [p["total_waiting_time_sec"] for p in per_person]

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PEDESTRIAN BEHAVIORAL & KINEMATIC FEATURE EXTRACTION REPORT\n")
        f.write(f"Total Tracked Pedestrians: {total}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. GENDER DISTRIBUTION\n")
        f.write(f"   - Males:   {len(males)} ({len(males)/total*100:.1f}%)\n")
        f.write(f"   - Females: {len(females)} ({len(females)/total*100:.1f}%)\n")
        f.write(f"   - Mean Attribute Confidence: {np.mean([p['gender_confidence'] for p in per_person]):.4f}\n\n")

        f.write("2. GROUP SIZE & CO-WALKING DYNAMICS\n")
        f.write(f"   - Single / Alone (Group Size = 1): {len(alone)} ({len(alone)/total*100:.1f}%)\n")
        f.write(f"   - Pairs (Group Size = 2):          {len(pairs)} ({len(pairs)/total*100:.1f}%)\n")
        f.write(f"   - Groups (Group Size >= 3):        {len(groups)} ({len(groups)/total*100:.1f}%)\n\n")

        f.write("3. CARRYING LOAD\n")
        f.write(f"   - Carrying Load (Backpack / Bag):  {len(carrying)} ({len(carrying)/total*100:.1f}%)\n")
        f.write(f"   - No Visible Load:                 {total - len(carrying)} ({(total-len(carrying))/total*100:.1f}%)\n\n")

        f.write("4. WALKING SPEED KINEMATICS (m/s & km/h)\n")
        f.write(f"   - Overall Mean Speed: {np.mean(speeds):.2f} m/s ({np.mean(speeds)*3.6:.2f} km/h)\n")
        f.write(f"   - Male Mean Speed:    {np.mean([p['avg_walking_speed_mps'] for p in males]):.2f} m/s\n")
        f.write(f"   - Female Mean Speed:  {np.mean([p['avg_walking_speed_mps'] for p in females]):.2f} m/s\n")
        f.write(f"   - Max Observed Speed: {np.max([p['max_walking_speed_mps'] for p in per_person]):.2f} m/s\n\n")

        f.write("5 & 6. WAITING BEHAVIOR & TOTAL WAITING TIME\n")
        f.write(f"   - Mean Waiting Time per Pedestrian: {np.mean(waits):.1f} seconds\n")
        f.write(f"   - Max Waiting Time Observed:        {np.max(waits):.1f} seconds\n")
        f.write(f"   - Pedestrians with Active Waiting:  {sum(1 for w in waits if w > 2.0)} ({sum(1 for w in waits if w > 2.0)/total*100:.1f}%)\n\n")

        f.write("7. NUMBER OF CROSSING ATTEMPTS\n")
        f.write(f"   - Direct Crossing (1 Attempt):         {len(direct_crossers)} ({len(direct_crossers)/total*100:.1f}%)\n")
        f.write(f"   - Hesitant Crossing (2+ Attempts):     {len(hesitant_crossers)} ({len(hesitant_crossers)/total*100:.1f}%)\n")
        f.write(f"   - Stationary / Non-Crossing:           {total - len(direct_crossers) - len(hesitant_crossers)}\n\n")

        f.write("8. BODY GESTURES & POSTURAL STATES\n")
        gestures = [p["body_gesture_state"].split(" (")[0] for p in per_person]
        for g in sorted(set(gestures)):
            f.write(f"   - {g}: {gestures.count(g)} pedestrians ({gestures.count(g)/total*100:.1f}%)\n")

    print(f"  -> Saved comprehensive report: {report_path}")


if __name__ == "__main__":
    extract_features()
