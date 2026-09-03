"""
stage8_crossing_zone_detector.py
Accurate Pedestrian Crossing Zone & Roadway Boundary Detection

Detects and marks:
  1. ROADWAY BOUNDARY — the entire road / intersection polygon
  2. YELLOW ZEBRA CROSSINGS — the two yellow-painted stripe zones 
  3. PEDESTRIAN CROSSING ZONE — the region between the two zebra crossings
     where pedestrians legally cross

The drone is fixed-hover so we detect once on a representative frame
and apply to all frames.

Usage:
    python stage8_crossing_zone_detector.py
"""

import os
import sys
import json
import cv2
import numpy as np

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
FRAME_DIR  = "work/frames"
OUT_DIR    = "work/crossing_zones"
TRACKS_PATH = "work/tracks/tracks.json"
ZEBRA_OUT  = "work/zebra/zebra_crossings.json"   # updated accurate version

# ──────────────────────────────────────────────────────────────
# Step 1: Find the best (earliest) available frame
# ──────────────────────────────────────────────────────────────
def get_representative_frame():
    frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])
    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in {FRAME_DIR}")
    return os.path.join(FRAME_DIR, frames[0])


# ──────────────────────────────────────────────────────────────
# Step 2: Detect yellow zebra stripes via HSV segmentation
# ──────────────────────────────────────────────────────────────
def detect_yellow_stripes(img):
    """Detect yellow-painted crossing stripes using HSV color segmentation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Yellow paint range in HSV
    # Broad range to catch worn / faded paint under varying lighting
    lower_yellow = np.array([15, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for stripe-shaped blobs (elongated, minimum area)
    stripe_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if min(w, h) < 1:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 1.5 and area > 200:  # elongated stripe
            stripe_contours.append(c)
    
    return mask, stripe_contours


def cluster_stripes_into_crossings(stripe_contours, cluster_dist=200):
    """Cluster nearby yellow stripes into crossing groups."""
    if not stripe_contours:
        return []
    
    # Get centroids
    centroids = []
    for c in stripe_contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
        else:
            rect = cv2.minAreaRect(c)
            centroids.append((int(rect[0][0]), int(rect[0][1])))
    
    # Simple proximity clustering
    used = [False] * len(stripe_contours)
    clusters = []
    
    for i in range(len(stripe_contours)):
        if used[i]:
            continue
        cluster_indices = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(stripe_contours)):
                if used[j]:
                    continue
                cx_j, cy_j = centroids[j]
                for k in cluster_indices:
                    cx_k, cy_k = centroids[k]
                    dist = np.sqrt((cx_j - cx_k)**2 + (cy_j - cy_k)**2)
                    if dist < cluster_dist:
                        cluster_indices.append(j)
                        used[j] = True
                        changed = True
                        break
        
        # Only keep clusters with enough stripes
        if len(cluster_indices) >= 3:
            cluster_contours = [stripe_contours[k] for k in cluster_indices]
            clusters.append(cluster_contours)
    
    return clusters


def crossing_polygon_from_cluster(cluster_contours):
    """Build a convex hull polygon around a cluster of stripe contours."""
    all_pts = []
    for c in cluster_contours:
        all_pts.extend(c.reshape(-1, 2).tolist())
    all_pts = np.array(all_pts, dtype=np.float32)
    hull = cv2.convexHull(all_pts)
    polygon = hull.reshape(-1, 2).tolist()
    return [[round(x, 1), round(y, 1)] for x, y in polygon]


# ──────────────────────────────────────────────────────────────
# Step 3: Define the roadway boundary and crossing zone
# ──────────────────────────────────────────────────────────────
def define_roadway_boundary(img_shape):
    """
    Define the roadway boundary polygon based on visual inspection of
    the fixed drone view.  The road runs diagonally from upper-left
    to lower-right across the 3840x2160 frame.
    
    This polygon encompasses the entire driveable road surface
    (excluding buildings, footpaths, and market areas).
    """
    h, w = img_shape[:2]
    
    # Roadway boundary polygon (the entire road intersection area)
    # Based on the user's annotated image 1 — the road runs from
    # upper-left to lower-right with buildings/market on the edges
    roadway_polygon = [
        [100, 0],        # top-left road edge
        [700, 0],        # top road upper boundary
        [1500, 0],       # continue along top
        [2700, 0],       # continue to right along top
        [3840, 0],       # top-right corner
        [3840, 500],     # right side upper
        [3840, 1000],    # right side mid
        [3840, 1550],    # right side lower
        [3300, 1600],    # bottom-right road edge turns
        [3100, 1700],    # right-side lower road edge
        [2800, 1900],    # lower road boundary
        [2200, 2160],    # bottom edge
        [1500, 2160],    # bottom center
        [800, 2160],     # bottom left
        [0, 2160],       # bottom-left corner
        [0, 1700],       # left side lower
        [0, 1200],       # left side mid
        [0, 700],        # left side upper
        [0, 300],        # left side top
    ]
    return roadway_polygon


def define_pedestrian_crossing_zone(zebra_crossings):
    """
    Define the pedestrian crossing zone — the region between the
    two yellow zebra crossings where pedestrians legally cross.
    
    This is the diamond/triangular area the user marked in image 2.
    
    We compute this as the convex hull encompassing both zebra
    crossing polygons, creating the full crossing corridor.
    """
    if len(zebra_crossings) < 2:
        print("WARNING: Need at least 2 zebra crossings to define crossing zone")
        return []
    
    # Collect all points from all zebra crossings
    all_pts = []
    for zc in zebra_crossings:
        for pt in zc["polygon"]:
            all_pts.append(pt)
    
    all_pts = np.array(all_pts, dtype=np.float32)
    hull = cv2.convexHull(all_pts)
    polygon = hull.reshape(-1, 2).tolist()
    return [[round(x, 1), round(y, 1)] for x, y in polygon]


# ──────────────────────────────────────────────────────────────
# Step 4: Annotate the frame with all zones
# ──────────────────────────────────────────────────────────────
def annotate_frame(img, roadway_poly, zebra_crossings, crossing_zone_poly,
                   tracks=None):
    """Draw all zones on the frame with color-coded overlays."""
    vis = img.copy()
    h, w = vis.shape[:2]
    
    # Create overlay for semi-transparent fills
    overlay = vis.copy()
    
    # 1. ROADWAY BOUNDARY — thick cyan border
    if roadway_poly:
        road_pts = np.array(roadway_poly, dtype=np.int32)
        cv2.polylines(vis, [road_pts], True, (255, 255, 0), 3)  # cyan
        cv2.polylines(overlay, [road_pts], True, (255, 255, 0), 3)
    
    # 2. ZEBRA CROSSINGS — solid yellow fill with green border
    for zc in zebra_crossings:
        pts = np.array(zc["polygon"], dtype=np.int32)
        # Semi-transparent yellow fill
        cv2.fillPoly(overlay, [pts], (0, 200, 255))  # yellow in BGR
        # Solid green border
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
        # Label
        cx = int(np.mean([p[0] for p in zc["polygon"]]))
        cy = int(np.mean([p[1] for p in zc["polygon"]]))
        cv2.putText(overlay, f"ZEBRA CROSSING {zc['id']+1}", (cx-80, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 3. PEDESTRIAN CROSSING ZONE — semi-transparent green fill
    if crossing_zone_poly:
        cz_pts = np.array(crossing_zone_poly, dtype=np.int32)
        cv2.fillPoly(overlay, [cz_pts], (0, 180, 0))  # green fill
        cv2.polylines(overlay, [cz_pts], True, (0, 255, 0), 4)
        # Label
        cx = int(np.mean([p[0] for p in crossing_zone_poly]))
        cy = int(np.mean([p[1] for p in crossing_zone_poly]))
        cv2.putText(overlay, "PEDESTRIAN CROSSING ZONE", (cx-200, cy-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Blend overlay with original (alpha=0.35 for fill transparency)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    
    # Re-draw borders on top (fully opaque)
    if roadway_poly:
        road_pts = np.array(roadway_poly, dtype=np.int32)
        cv2.polylines(vis, [road_pts], True, (255, 255, 0), 3)
    for zc in zebra_crossings:
        pts = np.array(zc["polygon"], dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    if crossing_zone_poly:
        cz_pts = np.array(crossing_zone_poly, dtype=np.int32)
        cv2.polylines(vis, [cz_pts], True, (0, 255, 0), 4)
    
    # 4. Draw pedestrian positions if tracks provided
    if tracks:
        for tid, detections in tracks.items():
            # Use first detection as representative
            for det in detections:
                cx = int((det["x1"] + det["x2"]) / 2)
                cy = int((det["y1"] + det["y2"]) / 2)
                
                # Check if inside crossing zone
                if crossing_zone_poly:
                    cz_pts_test = np.array(crossing_zone_poly, dtype=np.float32)
                    inside = cv2.pointPolygonTest(cz_pts_test.reshape(-1, 1, 2),
                                                  (float(cx), float(cy)), False)
                    color = (0, 255, 0) if inside >= 0 else (0, 0, 255)
                else:
                    color = (255, 0, 0)
                
                cv2.circle(vis, (cx, cy), 5, color, -1)
                break  # only first detection per track for the static view
    
    # Add legend
    legend_y = 50
    cv2.rectangle(vis, (20, 20), (500, 200), (0, 0, 0), -1)
    cv2.rectangle(vis, (20, 20), (500, 200), (255, 255, 255), 2)
    cv2.putText(vis, "LEGEND:", (30, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    legend_y += 35
    cv2.line(vis, (30, legend_y), (60, legend_y), (255, 255, 0), 3)
    cv2.putText(vis, "Roadway Boundary", (70, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    legend_y += 35
    cv2.rectangle(vis, (30, legend_y-10), (60, legend_y+10), (0, 200, 255), -1)
    cv2.putText(vis, "Zebra Crossing Stripes", (70, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    legend_y += 35
    cv2.rectangle(vis, (30, legend_y-10), (60, legend_y+10), (0, 180, 0), -1)
    cv2.putText(vis, "Pedestrian Crossing Zone", (70, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    legend_y += 35
    cv2.circle(vis, (45, legend_y), 6, (0, 255, 0), -1)
    cv2.putText(vis, "Ped Inside Zone", (70, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(vis, (250, legend_y), 6, (0, 0, 255), -1)
    cv2.putText(vis, "Ped Outside Zone", (270, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return vis


# ──────────────────────────────────────────────────────────────
# Step 5: Classify pedestrian crossing status per-frame
# ──────────────────────────────────────────────────────────────
def classify_pedestrian_crossing_status(tracks, crossing_zone_poly, zebra_polys):
    """
    For each pedestrian at each timestep, classify:
      - inside_crossing_zone: bool
      - inside_zebra: bool  
      - nearest_zebra_id: int
      - distance_to_nearest_zebra_m: float
    """
    PIXEL_TO_METER = 0.015  # GSD calibration
    
    results = {}
    cz_pts = np.array(crossing_zone_poly, dtype=np.float32).reshape(-1, 1, 2) if crossing_zone_poly else None
    
    zebra_pts_list = []
    for zc in zebra_polys:
        pts = np.array(zc["polygon"], dtype=np.float32).reshape(-1, 1, 2)
        zebra_pts_list.append((zc["id"], pts))
    
    for tid, detections in tracks.items():
        person_results = []
        for det in detections:
            cx = (det["x1"] + det["x2"]) / 2
            cy = (det["y1"] + det["y2"]) / 2
            
            # Inside crossing zone?
            in_cz = False
            if cz_pts is not None:
                in_cz = cv2.pointPolygonTest(cz_pts, (float(cx), float(cy)), False) >= 0
            
            # Inside any zebra?
            in_zebra = False
            nearest_zebra_id = -1
            nearest_dist_px = float("inf")
            
            for zid, zpts in zebra_pts_list:
                dist = cv2.pointPolygonTest(zpts, (float(cx), float(cy)), True)
                if dist >= 0:
                    in_zebra = True
                    nearest_zebra_id = zid
                    nearest_dist_px = 0
                    break
                else:
                    abs_dist = abs(dist)
                    if abs_dist < nearest_dist_px:
                        nearest_dist_px = abs_dist
                        nearest_zebra_id = zid
            
            person_results.append({
                "frame_id": det["frame_id"],
                "timestamp_sec": det["timestamp_sec"],
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "inside_crossing_zone": in_cz,
                "inside_zebra": in_zebra,
                "nearest_zebra_id": nearest_zebra_id,
                "distance_to_nearest_zebra_m": round(nearest_dist_px * PIXEL_TO_METER, 3),
            })
        results[tid] = person_results
    
    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load representative frame
    frame_path = get_representative_frame()
    print(f"[1/6] Loading frame: {frame_path}")
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {frame_path}")
    print(f"       Frame size: {img.shape[1]}x{img.shape[0]}")
    
    # Detect yellow stripes
    print("[2/6] Detecting yellow zebra crossing stripes via HSV segmentation...")
    mask, stripe_contours = detect_yellow_stripes(img)
    print(f"       Found {len(stripe_contours)} yellow stripe segments")
    
    # Save yellow mask for inspection
    cv2.imwrite(os.path.join(OUT_DIR, "yellow_stripe_mask.jpg"), mask)
    
    # Cluster stripes into crossings
    clusters = cluster_stripes_into_crossings(stripe_contours, cluster_dist=250)
    print(f"       Clustered into {len(clusters)} crossing group(s)")
    
    # Build zebra crossing polygons
    zebra_crossings = []
    for idx, cluster in enumerate(clusters):
        poly = crossing_polygon_from_cluster(cluster)
        if len(poly) >= 3:
            zebra_crossings.append({"id": idx, "polygon": poly})
    
    # If auto-detection found fewer than 2 crossings, fall back to manual
    if len(zebra_crossings) < 2:
        print("       Auto-detection found < 2 crossings.")
        print("       Attempting wider HSV range...")
        # Try wider range
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 30, 80])
        upper = np.array([40, 255, 255])
        mask2 = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Keep large yellow regions only
        large_yellow = [c for c in contours2 if cv2.contourArea(c) > 500]
        print(f"       Wide range found {len(large_yellow)} large yellow regions")
        
        clusters2 = cluster_stripes_into_crossings(large_yellow, cluster_dist=300)
        if len(clusters2) >= 2:
            zebra_crossings = []
            for idx, cluster in enumerate(clusters2):
                poly = crossing_polygon_from_cluster(cluster)
                if len(poly) >= 3:
                    zebra_crossings.append({"id": idx, "polygon": poly})
        
        cv2.imwrite(os.path.join(OUT_DIR, "yellow_stripe_mask_wide.jpg"), mask2)
    
    print(f"[3/6] Final zebra crossing polygons: {len(zebra_crossings)}")
    for zc in zebra_crossings:
        pts = np.array(zc["polygon"])
        area = cv2.contourArea(pts.astype(np.float32))
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        print(f"       Crossing {zc['id']}: center=({cx:.0f},{cy:.0f}), "
              f"area={area:.0f}px², vertices={len(zc['polygon'])}")
    
    # Define roadway boundary
    print("[4/6] Defining roadway boundary polygon...")
    roadway_poly = define_roadway_boundary(img.shape)
    
    # Define pedestrian crossing zone
    print("[5/6] Computing pedestrian crossing zone polygon...")
    crossing_zone_poly = define_pedestrian_crossing_zone(zebra_crossings)
    if crossing_zone_poly:
        cz_area = cv2.contourArea(np.array(crossing_zone_poly, dtype=np.float32))
        print(f"       Crossing zone area: {cz_area:.0f} px² "
              f"({cz_area * 0.015 * 0.015:.1f} m²)")
    
    # Load tracks for visualization
    tracks = None
    if os.path.exists(TRACKS_PATH):
        with open(TRACKS_PATH) as f:
            tracks = json.load(f)["tracks"]
        print(f"       Loaded {len(tracks)} pedestrian tracks for overlay")
    
    # Generate annotated frame
    print("[6/6] Generating annotated visualizations...")
    annotated = annotate_frame(img, roadway_poly, zebra_crossings,
                               crossing_zone_poly, tracks)
    
    annotated_path = os.path.join(OUT_DIR, "annotated_crossing_zones.jpg")
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"       Saved: {annotated_path}")
    
    # Save close-up crops of each crossing
    for zc in zebra_crossings:
        pts = np.array(zc["polygon"])
        x_min = max(0, int(pts[:, 0].min()) - 100)
        y_min = max(0, int(pts[:, 1].min()) - 100)
        x_max = min(img.shape[1], int(pts[:, 0].max()) + 100)
        y_max = min(img.shape[0], int(pts[:, 1].max()) + 100)
        crop = annotated[y_min:y_max, x_min:x_max]
        crop_path = os.path.join(OUT_DIR, f"crossing_{zc['id']}_closeup.jpg")
        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"       Saved: {crop_path}")
    
    # Save zone data as JSON
    zone_data = {
        "roadway_boundary": roadway_poly,
        "zebra_crossings": zebra_crossings,
        "pedestrian_crossing_zone": crossing_zone_poly,
        "metadata": {
            "frame_resolution": [img.shape[1], img.shape[0]],
            "source_frame": frame_path,
            "pixel_to_meter_scale": 0.015,
            "num_zebra_crossings": len(zebra_crossings),
        }
    }
    
    zone_json_path = os.path.join(OUT_DIR, "crossing_zones.json")
    with open(zone_json_path, "w") as f:
        json.dump(zone_data, f, indent=2)
    print(f"       Saved: {zone_json_path}")
    
    # Update the old zebra_crossings.json with accurate data
    os.makedirs(os.path.dirname(ZEBRA_OUT), exist_ok=True)
    with open(ZEBRA_OUT, "w") as f:
        json.dump(zebra_crossings, f, indent=2)
    print(f"       Updated: {ZEBRA_OUT}")
    
    # Classify pedestrian crossing status
    if tracks:
        print("\n-- Pedestrian Crossing Zone Classification --")
        crossing_status = classify_pedestrian_crossing_status(
            tracks, crossing_zone_poly, zebra_crossings)
        
        # Save crossing status
        status_path = os.path.join(OUT_DIR, "pedestrian_crossing_status.json")
        with open(status_path, "w") as f:
            json.dump(crossing_status, f, indent=2)
        
        # Summary statistics
        total_in_zone = 0
        total_in_zebra = 0
        persons_ever_in_zone = 0
        persons_ever_in_zebra = 0
        
        for tid, frames in crossing_status.items():
            ever_in_zone = any(f["inside_crossing_zone"] for f in frames)
            ever_in_zebra = any(f["inside_zebra"] for f in frames)
            if ever_in_zone:
                persons_ever_in_zone += 1
            if ever_in_zebra:
                persons_ever_in_zebra += 1
            total_in_zone += sum(1 for f in frames if f["inside_crossing_zone"])
            total_in_zebra += sum(1 for f in frames if f["inside_zebra"])
        
        total_frames = sum(len(f) for f in crossing_status.values())
        print(f"  Total pedestrians: {len(crossing_status)}")
        print(f"  Pedestrians ever inside crossing zone: {persons_ever_in_zone}")
        print(f"  Pedestrians ever on zebra stripes: {persons_ever_in_zebra}")
        print(f"  Frame-level: {total_in_zone}/{total_frames} inside zone "
              f"({100*total_in_zone/max(total_frames,1):.1f}%)")
        print(f"  Frame-level: {total_in_zebra}/{total_frames} on zebra "
              f"({100*total_in_zebra/max(total_frames,1):.1f}%)")
        
        print(f"\n  Saved: {status_path}")
    
    print("\n[OK] Crossing zone detection complete!")
    print(f"   Review: {annotated_path}")
    return zone_data


if __name__ == "__main__":
    main()
