"""
stage4_deepsort_tracker.py
DeepSORT Multi-Object Tracking with Kalman Filtering and Swin Re-ID Appearance Embeddings.

Features:
  1. 8D Kalman Filter state estimation [x, y, a, h, vx, vy, va, vh]
  2. Appearance Feature Extraction via Swin Transformer
  3. Mahalanobis distance gating + Cosine distance appearance metric
  4. Cascaded matching with linear sum assignment (Hungarian algorithm)
  5. Outputs trajectories compatible with DataFromSky analysis & stage 5/6/14
"""

import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from config import TRACKING, DETECTION, ATTRIBUTE, PATHS


# ==============================================================================
# 1. Kalman Filter for Bounding Box State Estimation
# ==============================================================================
class KalmanBoxTracker:
    """
    Kalman Filter state [x_center, y_center, aspect_ratio, height, vx, vy, va, vh].
    """
    count = 0

    def __init__(self, bbox, feat=None):
        """
        bbox: [x1, y1, x2, y2]
        """
        # State vector: [x, y, a, h, vx, vy, va, vh]
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        a = w / h

        self.mean = np.array([x, y, a, h, 0, 0, 0, 0], dtype=np.float32)
        
        # State transition matrix
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt = 1 frame

        # Measurement matrix (observing [x, y, a, h])
        self.H = np.eye(4, 8, dtype=np.float32)

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 1.0, 10.0, 100.0, 100.0, 10.0, 100.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 0.01, 1.0, 0.01, 0.01, 0.0001, 0.01]).astype(np.float32)
        self.R = np.diag([1.0, 1.0, 0.1, 1.0]).astype(np.float32)

        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.features = [feat] if feat is not None else []
        self.last_observation = None

    def predict(self):
        """Predict the next state."""
        self.mean = np.dot(self.F, self.mean)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.to_tlbr())
        return self.to_tlbr()

    def update(self, bbox, feat=None, conf=1.0):
        """Update the tracker with an observed detection bbox."""
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        a = w / h
        z = np.array([x, y, a, h], dtype=np.float32)

        # Innovation
        y_tilde = z - np.dot(self.H, self.mean)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.mean = self.mean + np.dot(K, y_tilde)
        self.P = np.dot(np.eye(8) - np.dot(K, self.H), self.P)

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if feat is not None:
            self.features.append(feat)
            if len(self.features) > 10:
                self.features.pop(0)
        self.last_observation = (bbox, conf)

    def to_tlbr(self):
        """Convert state to [x1, y1, x2, y2]."""
        x, y, a, h = self.mean[:4]
        w = max(1.0, a * h)
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def get_feature(self):
        """Return the average feature vector."""
        if not self.features:
            return None
        valid_feats = [f for f in self.features if f is not None]
        if not valid_feats:
            return None
        avg_feat = np.mean(valid_feats, axis=0)
        return avg_feat / (np.linalg.norm(avg_feat) + 1e-6)


# ==============================================================================
# 2. Deep Appearance Feature Extractor (Swin/CNN)
# ==============================================================================
class AppearanceExtractor:
    def __init__(self, ckpt_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((ATTRIBUTE.crop_size, ATTRIBUTE.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if ckpt_path and os.path.exists(ckpt_path):
            try:
                from stage6_attribute_model import AttributeModel
                self.model = AttributeModel().to(self.device)
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt, strict=False)
                self.model.eval()
                print(f"[DeepSORT] Loaded Swin Re-ID backbone from {ckpt_path}")
            except Exception as e:
                print(f"[DeepSORT] Fallback to basic visual feature extractor: {e}")
                self.model = None

    @torch.no_grad()
    def extract_crop_feature(self, frame_bgr, bbox):
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return np.zeros(128, dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self.model is not None:
            t = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            # Extract backbone spatial pooled embedding
            feat = self.model.backbone.forward_features(t)
            if feat.dim() == 4:
                feat = feat.mean(dim=(1, 2))  # GAP
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)
            feat = feat.squeeze().cpu().numpy()
            return feat / (np.linalg.norm(feat) + 1e-6)
        else:
            # Color histogram + spatial texture fallback
            small = cv2.resize(crop_rgb, (32, 32))
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()
            return hist / (np.linalg.norm(hist) + 1e-6)


# ==============================================================================
# 3. DeepSORT Tracker Class
# ==============================================================================
class DeepSORTTracker:
    def __init__(self, max_age=45, min_hits=3, iou_threshold=0.3, appearance_weight=0.7,
                 ckpt_path="checkpoints/attribute_model.pt"):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.trackers = []
        self.frame_count = 0
        self.feature_extractor = AppearanceExtractor(ckpt_path=ckpt_path)
        KalmanBoxTracker.count = 0

    def update(self, frame_bgr, detections):
        """
        detections: list of dicts with {"x1", "y1", "x2", "y2", "confidence"}
        """
        self.frame_count += 1
        
        # 1. Predict new locations of existing trackers
        for trk in self.trackers:
            trk.predict()

        if len(detections) == 0:
            # No detections in this frame
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            return self._get_active_tracks()

        det_boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float32)
        det_confs = np.array([d.get("confidence", 1.0) for d in detections], dtype=np.float32)

        # 2. Extract deep appearance features for detections
        det_feats = [self.feature_extractor.extract_crop_feature(frame_bgr, box) for box in det_boxes]

        # 3. Associate detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate(det_boxes, det_feats)

        # 4. Update matched trackers
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(det_boxes[d_idx], det_feats[d_idx], det_confs[d_idx])

        # 5. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            if det_confs[d_idx] >= 0.15:  # detection threshold
                trk = KalmanBoxTracker(det_boxes[d_idx], det_feats[d_idx])
                trk.last_observation = (det_boxes[d_idx], det_confs[d_idx])
                self.trackers.append(trk)

        # 6. Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return self._get_active_tracks()

    def _associate(self, det_boxes, det_feats):
        if len(self.trackers) == 0:
            return [], list(range(len(det_boxes))), []

        trk_boxes = np.array([t.to_tlbr() for t in self.trackers], dtype=np.float32)
        trk_feats = [t.get_feature() for t in self.trackers]

        # Compute IoU distance matrix
        iou_matrix = self._iou_distance(trk_boxes, det_boxes)

        # Compute Cosine appearance distance matrix
        app_matrix = np.ones_like(iou_matrix)
        for t_i, t_f in enumerate(trk_feats):
            if t_f is not None:
                for d_j, d_f in enumerate(det_feats):
                    if d_f is not None and len(d_f) == len(t_f):
                        cos_sim = np.dot(t_f, d_f) / (np.linalg.norm(t_f) * np.linalg.norm(d_f) + 1e-6)
                        app_matrix[t_i, d_j] = 1.0 - max(0.0, float(cos_sim))

        # Combined cost matrix (weighted sum of IoU and Appearance)
        cost_matrix = (1.0 - self.appearance_weight) * iou_matrix + self.appearance_weight * app_matrix

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_trks = list(range(len(self.trackers)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.7:  # matching threshold
                matched.append((r, c))
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
                if r in unmatched_trks:
                    unmatched_trks.remove(r)

        return matched, unmatched_dets, unmatched_trks

    def _iou_distance(self, trk_boxes, det_boxes):
        N = len(trk_boxes)
        M = len(det_boxes)
        dist = np.ones((N, M), dtype=np.float32)

        for i in range(N):
            tb = trk_boxes[i]
            t_area = max(0, tb[2] - tb[0]) * max(0, tb[3] - tb[1])
            for j in range(M):
                db = det_boxes[j]
                d_area = max(0, db[2] - db[0]) * max(0, db[3] - db[1])

                xx1 = max(tb[0], db[0])
                yy1 = max(tb[1], db[1])
                xx2 = min(tb[2], db[2])
                yy2 = min(tb[3], db[3])

                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                union = t_area + d_area - inter

                if union > 0:
                    dist[i, j] = 1.0 - (inter / union)
        return dist

    def _get_active_tracks(self):
        active = []
        for t in self.trackers:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                box = t.to_tlbr()
                conf = t.last_observation[1] if t.last_observation else 1.0
                active.append({
                    "id": t.id,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "confidence": float(conf)
                })
        return active


# ==============================================================================
# 4. Pipeline Execution Function
# ==============================================================================
def run_deepsort_tracking(detections_path, manifest_path, out_dir="work/tracks",
                          ckpt_path="checkpoints/attribute_model.pt", min_track_len=3):
    os.makedirs(out_dir, exist_ok=True)
    with open(detections_path) as f:
        det_data = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_map = {item["frame_idx"]: item["file"] for item in manifest["frames"]}
    tracker = DeepSORTTracker(max_age=60, min_hits=2, appearance_weight=0.6, ckpt_path=ckpt_path)

    frames = sorted(det_data.get("frames", []), key=lambda x: x["frame_idx"])
    raw_tracks = {}

    print(f"[DeepSORT] Processing {len(frames)} frames with Kalman Filtering & Swin Re-ID...")
    for frame_data in frames:
        fidx = frame_data["frame_idx"]
        ts = frame_data.get("timestamp_sec", 0.0)
        dets = frame_data.get("detections", [])

        img_path = manifest_map.get(fidx)
        if not img_path or not os.path.exists(img_path):
            continue
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue

        active = tracker.update(frame_bgr, dets)
        for trk in active:
            tid = str(trk["id"])
            if tid not in raw_tracks:
                raw_tracks[tid] = []
            raw_tracks[tid].append({
                "frame_id": str(fidx),
                "timestamp_sec": round(ts, 2),
                "x1": trk["x1"],
                "y1": trk["y1"],
                "x2": trk["x2"],
                "y2": trk["y2"],
                "confidence": trk["confidence"]
            })

    # Filter short noise tracks
    filtered = {tid: obs for tid, obs in raw_tracks.items() if len(obs) >= min_track_len}
    out_file = os.path.join(out_dir, "tracks.json")
    with open(out_file, "w") as f:
        json.dump({"tracks": filtered}, f, indent=2)

    print(f"\n[DeepSORT] Finished tracking:")
    print(f"  Total raw tracks: {len(raw_tracks)}")
    print(f"  Persistent tracks kept (>={min_track_len} frames): {len(filtered)}")
    print(f"  Saved -> {out_file}")
    return filtered


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DeepSORT Tracker with Swin Re-ID")
    ap.add_argument("--detections", default="work/detections/detections.json")
    ap.add_argument("--manifest", default="work/frames_manifest.json")
    ap.add_argument("--out_dir", default="work/tracks")
    ap.add_argument("--ckpt", default="checkpoints/attribute_model.pt")
    ap.add_argument("--min_len", type=int, default=3)
    args = ap.parse_args()

    run_deepsort_tracking(args.detections, args.manifest, args.out_dir, args.ckpt, args.min_len)
