import argparse
import json
import os
import cv2
import numpy as np

from config import TRACKING, PATHS

def compute_feature(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return resized.flatten().astype(np.float32)

def is_blurry(crop, threshold=50.0):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold

def farthest_point_sampling(features, k):
    n = len(features)
    if n <= k:
        return list(range(n))
    
    selected_indices = [0]
    distances = np.linalg.norm(features - features[0], axis=1)
    
    for _ in range(1, k):
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)
        new_dist = np.linalg.norm(features - features[next_idx], axis=1)
        distances = np.minimum(distances, new_dist)
        
    return selected_indices

def process_person(person_id, track_list, manifest_map, out_dir, k=8, blur_threshold=50.0):
    person_dir = os.path.join(out_dir, f"person_{person_id}")
    os.makedirs(person_dir, exist_ok=True)
    
    valid_crops = []
    valid_features = []
    valid_meta = []
    
    discarded = 0
    
    for det in track_list:
        frame_idx = det["frame_id"]
        if frame_idx not in manifest_map:
            continue
            
        fpath = manifest_map[frame_idx]
        frame = cv2.imread(fpath)
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        x1 = max(0, int(det["x1"]))
        y1 = max(0, int(det["y1"]))
        x2 = min(w, int(det["x2"]))
        y2 = min(h, int(det["y2"]))
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
            
        crop = frame[y1:y2, x1:x2]
        
        if is_blurry(crop, blur_threshold):
            discarded += 1
            continue
            
        feature = compute_feature(crop)
        valid_crops.append(crop)
        valid_features.append(feature)
        valid_meta.append(det)
        
    if not valid_crops:
        return [], discarded
        
    features = np.array(valid_features)
    selected_indices = farthest_point_sampling(features, k)
    
    results = []
    for i, idx in enumerate(sorted(selected_indices)):
        crop = valid_crops[idx]
        meta = valid_meta[idx]
        
        crop_name = f"crop_{i:02d}.jpg"
        crop_path = os.path.join(person_dir, crop_name)
        cv2.imwrite(crop_path, crop)
        
        results.append({
            "crop_file": os.path.normpath(crop_path).replace("\\", "/"),
            "frame_id": meta["frame_id"],
            "timestamp_sec": meta["timestamp_sec"],
            "bbox": [meta["x1"], meta["y1"], meta["x2"], meta["y2"]],
            "confidence": meta["confidence"]
        })
        
    return results, discarded

def main():
    ap = argparse.ArgumentParser(description="Section 5 close / 6.1 Diverse Crop Selection")
    ap.add_argument("--tracks", default="work/tracks/tracks.json")
    ap.add_argument("--manifest", default="work/frames_manifest.json")
    ap.add_argument("--out_dir", default="work/crops")
    ap.add_argument("--k", type=int, default=TRACKING.diverse_crops_per_id)
    args = ap.parse_args()
    
    with open(args.tracks, "r") as f:
        tracks_data = json.load(f)
        
    with open(args.manifest, "r") as f:
        manifest_data = json.load(f)
        
    manifest_map = {}
    for item in manifest_data["frames"]:
        manifest_map[item["frame_idx"]] = item["file"]
        manifest_map[str(item["frame_idx"])] = item["file"]
    
    out_index = {"persons": {}}
    total_persons = len(tracks_data.get("tracks", {}))
    total_discarded = 0
    total_selected = 0
    
    for person_id, track_list in tracks_data.get("tracks", {}).items():
        results, discarded = process_person(person_id, track_list, manifest_map, args.out_dir, k=args.k)
        total_discarded += discarded
        if results:
            out_index["persons"][person_id] = results
            total_selected += len(results)
            
    index_path = os.path.join(args.out_dir, "crop_index.json")
    with open(index_path, "w") as f:
        json.dump(out_index, f, indent=2)
        
    print(f"Summary Stats:")
    print(f"Total persons processed: {total_persons}")
    print(f"Total crops selected: {total_selected} (avg {total_selected/max(1, len(out_index['persons'])):.1f} per valid person)")
    print(f"Total crops discarded for blur: {total_discarded}")
    print(f"Index saved to: {index_path}")

if __name__ == "__main__":
    main()
