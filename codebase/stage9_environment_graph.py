"""
stage9_environment_graph.py
Section III-C — Pedestrian-Centric Environment Graph (Eq. 1-6)

IMPORTANT — sampling-rate mismatch with the base pipeline:
stage2's frame sampling (1 frame / 1-2s) was designed for efficient
gender/attribute annotation COVERAGE across a 30-min video. The crossing-
intention framework needs the OPPOSITE: densely-spaced frames within a
short K1=1s window (Section IV-A), so velocity/acceleration (Eq. 3-4)
are meaningful. So this module does its OWN dense, targeted re-extraction
from the raw video for each K1-second window it needs — it does not
reuse stage2's sparse frames.

Pipeline per pedestrian-centric sample (one K1-second window, ending
K2 seconds before a labelled crossing, or a random window for negatives):
  1. extract_dense_window()   — K_history_frames evenly spaced frames,
                                 pulled directly from the raw video via ffmpeg
  2. detect entities in each frame (person + vehicle classes + traffic
     light) via stage3's RT-DETR, COCO_ENV_CLASSES
  3. match_consecutive()      — greedy nearest-centroid matching across
                                 the K frames to get short local motion
                                 tracklets for surrounding entities (the
                                 OBSERVED pedestrian instead uses its real
                                 ByteTrack trajectory from stage4)
  4. build_appearance_subgraph() — G^t_A, Eq. 1
  5. build_motion_subgraph()     — G^t_M, Eq. 1-6

Output: one .json + .npy bundle per sample under work/env_graphs/,
consumed by stage10 (environment encoder) and stage12 (decoder/dataset).

Usage:
    python stage9_environment_graph.py build --video drone.mp4 \
        --tracks work/tracks/tracks.json --zebra work/zebra/zebra_crossings.json \
        --pedestrian_track_id 17 --center_time_sec 184.5 --out_dir work/env_graphs
"""

import argparse
import json
import os
import subprocess
import cv2
import numpy as np

from config import INTENTION, DETECTION


# ----------------------------------------------------------------------
# Eq. 1 — kernel function for edge weights (shared by both sub-graphs)
# ----------------------------------------------------------------------
def kernel_function(pos_i, pos_j):
    """a^t_ij = 1/||pos_i - pos_j||_2 if distance != 0, else 0."""
    d = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
    return 0.0 if d == 0 else 1.0 / d


def build_adjacency(positions):
    """positions: list of (x,y). Returns (n,n) weighted adjacency via Eq. 1."""
    n = len(positions)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = kernel_function(positions[i], positions[j])
    return A


# ----------------------------------------------------------------------
# Dense window extraction (ffmpeg-based, accurate short-window seeking)
# ----------------------------------------------------------------------
def extract_dense_window(video_path, t_start, t_end, k_frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timestamps = np.linspace(t_start, t_end, k_frames).tolist()
    files = []
    
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        for i, t in enumerate(timestamps):
            out_path = os.path.join(out_dir, f"dense_{i:02d}_t{t:08.3f}.jpg")
            if os.path.exists(out_path):
                files.append({"file": out_path, "timestamp_sec": round(t, 3)})
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.imwrite(out_path, frame)
                files.append({"file": out_path, "timestamp_sec": round(t, 3)})
        cap.release()
    return files


# ----------------------------------------------------------------------
# Consecutive-frame matching for surrounding entities (no persistent
# long-term ID needed here — only local motion within the K-frame window)
# ----------------------------------------------------------------------
def match_consecutive(prev_entities, curr_entities, max_dist_px=80):
    """Greedy nearest-centroid matching. Each entity dict has x1,y1,x2,y2.
    Returns list of (prev_idx, curr_idx) pairs."""
    if not prev_entities or not curr_entities:
        return []
    prev_centers = [((e["x1"] + e["x2"]) / 2, e["y2"]) for e in prev_entities]  # bottom-center
    curr_centers = [((e["x1"] + e["x2"]) / 2, e["y2"]) for e in curr_entities]

    pairs = []
    used_curr = set()
    for i, pc in enumerate(prev_centers):
        best_j, best_d = None, max_dist_px
        for j, cc in enumerate(curr_centers):
            if j in used_curr:
                continue
            d = np.hypot(pc[0] - cc[0], pc[1] - cc[1])
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            pairs.append((i, best_j))
            used_curr.add(best_j)
    return pairs


# ----------------------------------------------------------------------
# Eq. 2-6 — motion + position features for the motion sub-graph
# ----------------------------------------------------------------------
def bottom_center(box):
    return ((box["x1"] + box["x2"]) / 2, box["y2"])  # Eq. 2, l/r -> bottom-center


def point_to_polygon_distance(pt, polygon):
    """Eq. 6 (generalized to an arbitrary polygon rather than only an
    axis-aligned box): shortest distance from a point to the polygon
    boundary/interior."""
    poly = np.array(polygon, dtype=np.float32)
    p = np.array(pt, dtype=np.float32)
    # inside check (ray casting) -> distance 0 if inside
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > p[1]) != (yj > p[1])) and \
           (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    if inside:
        return 0.0
    # min distance to each edge segment
    min_d = float("inf")
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        proj = a + t * ab
        min_d = min(min_d, float(np.linalg.norm(p - proj)))
    return min_d


def nearest_zebra_distance(pt, zebra_polygons):
    if not zebra_polygons:
        return 0.0  # no zebra crossing known nearby -> treat as not meaningfully far (padding)
    return min(point_to_polygon_distance(pt, z["polygon"]) for z in zebra_polygons)


def motion_features_for_entity(history_boxes, zebra_polygons, category):
    """history_boxes: list of up to 3 consecutive {x1,y1,x2,y2} for ONE
    entity (oldest -> newest), already matched/tracked across frames.
    category: 'pedestrian' | 'vehicle' | 'zebra_crossing'
    Returns the 13-dim feature vector: [cat(3), motion(5), position(5)]
    exactly as defined in Section III-C.2."""
    cat_onehot = {
        "pedestrian": [1, 0, 0], "vehicle": [0, 1, 0], "zebra_crossing": [0, 0, 1],
    }[category]

    if len(history_boxes) == 0:
        return np.array(cat_onehot + [0] * 10, dtype=np.float32)

    cur = history_boxes[-1]
    cur_lm = bottom_center(cur)  # Eq. 2

    if len(history_boxes) >= 2:
        prev = history_boxes[-2]
        prev_lm = bottom_center(prev)
        vx = cur_lm[0] - prev_lm[0]
        vy = cur_lm[1] - prev_lm[1]
    else:
        vx, vy = 0.0, 0.0

    if len(history_boxes) >= 3:
        prev2 = history_boxes[-3]
        prev_lm2 = bottom_center(prev2)
        prev_v = history_boxes[-2]
        prev_v_lm = bottom_center(prev_v)
        vx_prev = prev_v_lm[0] - prev_lm2[0]
        vy_prev = prev_v_lm[1] - prev_lm2[1]
        ax = vx - vx_prev
        ay = vy - vy_prev
    else:
        ax, ay = 0.0, 0.0

    if len(history_boxes) >= 2 and (cur_lm[0] - prev_lm[0]) != 0:
        alpha = float(np.arctan2(cur_lm[1] - prev_lm[1], cur_lm[0] - prev_lm[0]))
    else:
        alpha = 0.0

    d_zebra = nearest_zebra_distance(cur_lm, zebra_polygons)

    motion = [vx, vy, ax, ay, alpha]
    position = [cur["x1"], cur["y1"], cur["x2"], cur["y2"], d_zebra]
    return np.array(cat_onehot + motion + position, dtype=np.float32)


# ----------------------------------------------------------------------
# Graph builders
# ----------------------------------------------------------------------
def build_appearance_subgraph(frame_path, pedestrian_box, nearby_vehicles, nearby_pedestrians,
                               nearby_lights, cnn_extractor,
                               M=INTENTION.M_vehicles, N=INTENTION.N_pedestrians,
                               L=INTENTION.L_traffic_lights):
    """G^t_A: appearance sub-graph. Nodes = observed pedestrian + M vehicles
    + N pedestrians + L traffic lights (zero-padded if fewer present), Eq. 1
    weighted adjacency from bbox-center distances."""
    import cv2

    def crop_feat(img, box):
        x1, y1, x2, y2 = [int(max(v, 0)) for v in (box["x1"], box["y1"], box["x2"], box["y2"])]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(INTENTION.appearance_feat_dim, dtype=np.float32)
        return cnn_extractor(crop)

    img = cv2.imread(frame_path)
    nodes = [pedestrian_box] + nearby_vehicles[:M] + nearby_pedestrians[:N] + nearby_lights[:L]
    # zero-pad to fixed size (M+N+L+1) so batches are uniform, per the paper's own padding note
    pad_count = (M + N + L + 1) - len(nodes)
    positions = [bottom_center(b) for b in nodes] + [(0.0, 0.0)] * pad_count
    feats = [crop_feat(img, b) for b in nodes] + \
            [np.zeros(INTENTION.appearance_feat_dim, dtype=np.float32)] * pad_count

    A = build_adjacency(positions)
    X = np.stack(feats, axis=0)
    return {"A": A, "X": X, "num_real_nodes": len(nodes)}


def build_motion_subgraph(pedestrian_history, nearby_vehicle_tracklets, nearby_pedestrian_tracklets,
                           zebra_polygons, M=INTENTION.M_vehicles, N=INTENTION.N_pedestrians,
                           Z=INTENTION.Z_zebra_crossings):
    """G^t_M: motion sub-graph at the CURRENT (last) timestep of the window.
    pedestrian_history / *_tracklets: list of up to 3 consecutive boxes
    (oldest->newest) per entity."""
    entities = [("pedestrian", pedestrian_history)] + \
               [("vehicle", t) for t in nearby_vehicle_tracklets[:M]] + \
               [("pedestrian", t) for t in nearby_pedestrian_tracklets[:N]]
    zebra_entries = [("zebra_crossing", z["polygon"]) for z in zebra_polygons[:Z]]

    positions, feats = [], []
    for cat, hist in entities:
        cur_box = hist[-1] if hist else {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        positions.append(bottom_center(cur_box))
        feats.append(motion_features_for_entity(hist, zebra_polygons, cat))
    for cat, polygon in zebra_entries:
        centroid = np.mean(np.array(polygon), axis=0).tolist()
        positions.append(tuple(centroid))
        xs, ys = zip(*polygon)
        pseudo_box = {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
        feats.append(motion_features_for_entity([pseudo_box], zebra_polygons, cat))

    total_nodes = 1 + M + N + Z
    pad_count = total_nodes - len(positions)
    positions += [(0.0, 0.0)] * pad_count
    feats += [np.zeros(13, dtype=np.float32)] * pad_count

    A = build_adjacency(positions)
    X = np.stack(feats, axis=0)
    return {"A": A, "X": X, "num_real_nodes": len(entities) + len(zebra_entries)}


def main():
    ap = argparse.ArgumentParser(description="Section III-C — environment graph construction "
                                              "(requires torch/ultralytics/timm at runtime for "
                                              "the CNN feature extractor; the geometry/motion math "
                                              "above has no such dependency).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--zebra", default="work/zebra/zebra_crossings.json")
    ap.add_argument("--out_dir", default="work/env_graphs")
    ap.add_argument("--center_time_sec", type=float, required=True)
    ap.add_argument("--pedestrian_box", required=True, help='JSON: {"x1":..,"y1":..,"x2":..,"y2":..}')
    ap.add_argument("--sample_id", required=True)
    args = ap.parse_args()

    with open(args.zebra) as f:
        zebra_polygons = json.load(f)

    t0 = args.center_time_sec - INTENTION.K1_clip_duration_sec
    t1 = args.center_time_sec
    dense_dir = os.path.join(args.out_dir, f"{args.sample_id}_frames")
    frames = extract_dense_window(args.video, t0, t1, INTENTION.K_history_frames, dense_dir)
    print(f"Extracted {len(frames)} dense frames for sample {args.sample_id} "
          f"in window [{t0:.2f}, {t1:.2f}]s.")
    print("Run stage3's detector (COCO_ENV_CLASSES) on these frames, then feed the boxes "
          "to build_appearance_subgraph()/build_motion_subgraph() — see stage12's dataset "
          "builder for the full orchestration used during training.")


if __name__ == "__main__":
    main()
