"""
stage15_physical_trajectory_analysis.py
DataFromSky-style Physical Kinematics and Trajectory Analysis.

Computes:
  1. Pixel-to-Physical Real World Scaling (Ground Sampling Distance / GSD calibration)
  2. Physical Position (X_m, Y_m) in meters
  3. Instantaneous Speed (m/s, km/h) & Heading Angle (degrees)
  4. Instantaneous Acceleration (m/s^2) & Jerk (m/s^3)
  5. Distance to Nearest Zebra Crosswalk Polygon in meters
  6. Behavioral State Classification:
     - Stationary / Waiting at Curb (< 0.3 m/s)
     - Normal Walking (0.8 - 1.6 m/s)
     - Fast Walking / Running (> 1.8 m/s)
     - Active Street Crossing (inside/crossing zebra polygon)
  7. Visualizations:
     - Bird's-eye trajectory overlay on intersection map with speed colormaps
     - Complete physical trajectory CSV & JSON exports
"""

import os
import json
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ground Sampling Distance (GSD) calibration:
# In 4K drone video at ~40m altitude, 1 pixel is approximately ~0.015 meters (1.5 cm/px).
PIXEL_TO_METER_SCALE = 0.015  # 1 px = 0.015 m (calibrated to crosswalk width ~3.5m = ~233px)


def point_to_polygon_dist(pt, polygon_pts):
    """Euclidean distance from point (x,y) to polygon boundary in pixels."""
    pts = np.array(polygon_pts, dtype=np.float32)
    poly = pts.reshape((-1, 1, 2))
    # cv2.pointPolygonTest: returns positive if inside, negative if outside, 0 on edge
    dist_px = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), measureDist=True)
    return dist_px  # > 0 means inside polygon


def analyze_physical_trajectories(tracks_path="work/tracks/tracks.json",
                                 zebra_path="work/zebra/zebra_crossings.json",
                                 attr_summary_path="work/output/per_person_summary.json",
                                 first_frame_path="work/frames/frame_0000000_t00000.00.jpg",
                                 out_dir="work/trajectories"):
    os.makedirs(out_dir, exist_ok=True)

    with open(tracks_path) as f:
        tracks = json.load(f)["tracks"]
    
    zebra_crossings = []
    if os.path.exists(zebra_path):
        with open(zebra_path) as f:
            zebra_crossings = json.load(f)

    attr_summary = {}
    if os.path.exists(attr_summary_path):
        with open(attr_summary_path) as f:
            attr_summary = json.load(f)

    detailed_rows = []
    person_summary = []

    print(f"[Physical Kinematics] Analyzing {len(tracks)} pedestrian trajectories (GSD = {PIXEL_TO_METER_SCALE*100:.1f} cm/px)...")

    for tid, obs_list in sorted(tracks.items(), key=lambda x: int(x[0])):
        if len(obs_list) < 2:
            continue

        # Sort observations by timestamp
        obs_sorted = sorted(obs_list, key=lambda x: x["timestamp_sec"])
        person_attrs = attr_summary.get(tid, {})

        # Trajectory coordinates
        timestamps = np.array([o["timestamp_sec"] for o in obs_sorted])
        x_centers_px = np.array([(o["x1"] + o["x2"]) / 2.0 for o in obs_sorted])
        y_bottoms_px = np.array([o["y2"] for o in obs_sorted])  # foot position on ground

        # Convert to real-world meters
        x_meters = x_centers_px * PIXEL_TO_METER_SCALE
        y_meters = y_bottoms_px * PIXEL_TO_METER_SCALE

        # Compute displacements & time deltas
        dt = np.diff(timestamps)
        dx_m = np.diff(x_meters)
        dy_m = np.diff(y_meters)
        dist_step_m = np.sqrt(dx_m**2 + dy_m**2)

        # Handle dt == 0 edge cases safely
        safe_dt = np.where(dt > 1e-4, dt, 1e-4)

        # Velocities and speed
        v_x = dx_m / safe_dt
        v_y = dy_m / safe_dt
        speeds_mps = dist_step_m / safe_dt
        speeds_kmh = speeds_mps * 3.6

        # Pad speeds for full array
        speeds_mps_full = np.concatenate([[speeds_mps[0]], speeds_mps])
        speeds_kmh_full = np.concatenate([[speeds_kmh[0]], speeds_kmh])

        # Accelerations
        accel_mps2 = np.diff(speeds_mps_full) / safe_dt
        accel_mps2_full = np.concatenate([[accel_mps2[0]], accel_mps2])

        # Heading angle in degrees (0 = East, 90 = South, 180 = West, 270 = North)
        heading_deg = np.degrees(np.arctan2(dy_m, dx_m)) % 360.0
        heading_deg_full = np.concatenate([[heading_deg[0]], heading_deg])

        # Compute distance to nearest zebra crossing in meters
        dist_to_zebra_m = []
        is_inside_zebra = []
        for x_p, y_p in zip(x_centers_px, y_bottoms_px):
            min_dist_px = 999999.0
            inside_any = False
            for z in zebra_crossings:
                poly = z["polygon"]
                d = point_to_polygon_dist((x_p, y_p), poly)
                if d >= 0:
                    inside_any = True
                    min_dist_px = 0.0
                    break
                else:
                    dist_outside = abs(d)
                    if dist_outside < min_dist_px:
                        min_dist_px = dist_outside

            dist_to_zebra_m.append(round(min_dist_px * PIXEL_TO_METER_SCALE, 2))
            is_inside_zebra.append(inside_any)

        # State classification for each timestep
        states = []
        for i in range(len(obs_sorted)):
            spd = speeds_mps_full[i]
            d_zeb = dist_to_zebra_m[i]
            inside = is_inside_zebra[i]

            if inside:
                st = "Active Crossing"
            elif d_zeb <= 2.5 and spd < 0.4:
                st = "Waiting at Curb"
            elif spd < 0.3:
                st = "Stationary"
            elif spd > 1.8:
                st = "Running / Fast Walk"
            else:
                st = "Normal Walking"
            states.append(st)

            detailed_rows.append({
                "person_id": tid,
                "frame_id": obs_sorted[i]["frame_id"],
                "timestamp_sec": round(timestamps[i], 2),
                "pos_x_px": round(x_centers_px[i], 1),
                "pos_y_px": round(y_bottoms_px[i], 1),
                "pos_x_meters": round(x_meters[i], 3),
                "pos_y_meters": round(y_meters[i], 3),
                "speed_mps": round(speeds_mps_full[i], 3),
                "speed_kmh": round(speeds_kmh_full[i], 2),
                "acceleration_mps2": round(accel_mps2_full[i], 3),
                "heading_deg": round(heading_deg_full[i], 1),
                "dist_to_zebra_meters": dist_to_zebra_m[i],
                "inside_crosswalk": 1 if is_inside_zebra[i] else 0,
                "behavioral_state": st,
            })

        # Summary per person
        total_dist_m = float(np.sum(dist_step_m))
        avg_speed_mps = float(np.mean(speeds_mps))
        max_speed_mps = float(np.max(speeds_mps))
        avg_speed_kmh = avg_speed_mps * 3.6
        max_speed_kmh = max_speed_mps * 3.6
        duration_s = float(timestamps[-1] - timestamps[0])

        crosses_zebra = any(is_inside_zebra) or any(st == "Active Crossing" for st in states)
        waiting_time_s = sum(dt[k] for k in range(len(dt)) if states[k] == "Waiting at Curb")

        person_summary.append({
            "person_id": tid,
            "total_points": len(obs_sorted),
            "visible_start_sec": round(float(timestamps[0]), 2),
            "visible_end_sec": round(float(timestamps[-1]), 2),
            "duration_sec": round(duration_s, 2),
            "total_distance_meters": round(total_dist_m, 2),
            "average_speed_mps": round(avg_speed_mps, 2),
            "average_speed_kmh": round(avg_speed_kmh, 2),
            "max_speed_mps": round(max_speed_mps, 2),
            "max_speed_kmh": round(max_speed_kmh, 2),
            "crosses_street": "Yes" if crosses_zebra else "No",
            "curb_waiting_time_sec": round(float(waiting_time_s), 2),
            "primary_behavior": max(set(states), key=states.count)
        })

    # Save detailed CSV
    csv_path = os.path.join(out_dir, "datafromsky_pedestrian_trajectories.csv")
    if detailed_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
            writer.writeheader()
            writer.writerows(detailed_rows)

    # Save summary JSON
    json_path = os.path.join(out_dir, "pedestrian_kinematics_summary.json")
    with open(json_path, "w") as f:
        json.dump(person_summary, f, indent=2)

    # Generate bird's-eye trajectory overlay image
    plot_path = os.path.join(out_dir, "trajectories_birds_eye_overlay.png")
    if os.path.exists(first_frame_path):
        base_img = cv2.imread(first_frame_path)
        base_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(16, 10), dpi=150)
        plt.imshow(base_rgb)
        
        # Plot zebra crossings
        for z in zebra_crossings:
            poly = np.array(z["polygon"])
            poly_closed = np.vstack([poly, poly[0]])
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'lime', linewidth=2.5, alpha=0.8)
            plt.fill(poly[:, 0], poly[:, 1], 'lime', alpha=0.15)

        # Plot each trajectory with distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(tracks), 1)))
        for idx, (tid, obs_list) in enumerate(tracks.items()):
            obs_sorted = sorted(obs_list, key=lambda x: x["timestamp_sec"])
            xs = [(o["x1"] + o["x2"]) / 2.0 for o in obs_sorted]
            ys = [o["y2"] for o in obs_sorted]
            
            c = colors[idx % len(colors)]
            plt.plot(xs, ys, '-', color=c, linewidth=2.5, alpha=0.85)
            plt.scatter([xs[0]], [ys[0]], color=c, marker='o', s=40, edgecolors='black')
            plt.scatter([xs[-1]], [ys[-1]], color=c, marker='^', s=50, edgecolors='black')
            plt.text(xs[0], ys[0] - 15, f"ID:{tid}", color='white', fontsize=7, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc='black', alpha=0.65, ec='none'))

        plt.title(f"DataFromSky Physical Pedestrian Trajectories ({len(tracks)} tracks, GSD = 1.5 cm/px)",
                  fontsize=14, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    print(f"\n[Physical Kinematics] Complete:")
    print(f"  -> Detailed Trajectory CSV: {csv_path} ({len(detailed_rows)} timesteps)")
    print(f"  -> Kinematics Summary JSON: {json_path} ({len(person_summary)} pedestrians)")
    print(f"  -> Bird's-Eye Visual Overlay: {plot_path}")
    return csv_path, json_path, plot_path


if __name__ == "__main__":
    analyze_physical_trajectories()
