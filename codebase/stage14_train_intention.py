"""
stage14_train_intention.py
Orchestrates stages 8-12 into one trainable crossing-intention pipeline.

Two-phase training, matching the paper's own description:
  PHASE 1 (unsupervised pretraining):
    - GCNAutoencoder on appearance sub-graphs collected from many frames
      across the video (Eq. 7-8)
    - TrajectoryLSTMAutoencoder on all pedestrian trajectories from
      stage4's tracks.json (Eq. 10)
    (Skeleton extractor: the paper pretrains on Kinetics-400; we don't
    replicate that here — see README's substitution note. It trains from
    random init as part of Phase 2 instead, which will need more labeled
    samples to converge than the paper's setup assumed.)

  PHASE 2 (end-to-end supervised training on labelled crossing samples):
    For each (track_id, window, label) sample from stage12's dataset
    builder: extract dense frames -> detect entities -> build per-frame
    appearance+motion sub-graphs -> encode through the (now-frozen)
    pretrained GCN-AE + appearance temporal LSTM -> ST-GCNN on motion
    graphs -> pedestrian's own appearance/trajectory/skeleton features ->
    intention decoder -> BCE loss.

>>> REALISTIC TIMING CHECK <<<
The paper's own reported convergence time for a similar-scope framework
was 5 days 16 hours on a single RTX 3090 (Section IV-B). This module is
comparable in scope. On a free/Pro Colab T4-class GPU with a few hundred
labelled samples (realistic for one 30-min video), expect Phase 2 to take
hours, not minutes — each sample requires its own multi-class detector
pass + pose estimation over K dense frames, which is the actual
bottleneck, not the decoder's tiny LSTM. Plan accordingly: this is not a
"run it and check back in 20 minutes" step.

Usage:
    python stage14_train_intention.py --video drone.mp4 \
        --tracks work/tracks/tracks.json \
        --zebra work/zebra/zebra_crossings.json \
        --crossing_labels crossing_labels.json \
        --out_dir work/intention
"""

import argparse
import json
import os

import numpy as np
import torch

from config import INTENTION, DETECTION
import stage9_environment_graph as env_graph
from stage10_environment_encoder import GCNAutoencoder, AppearanceTemporalLSTM, STGCNN, \
    train_gcn_autoencoder
from stage11_pedestrian_state_encoder import AppearanceCNN, TrajectoryLSTMAutoencoder, \
    SkeletonGCNExtractor, skeleton_node_features, train_trajectory_autoencoder
from stage12_intention_decoder import IntentionPredictionDecoder, build_crossing_samples, \
    split_samples_by_segment, evaluate_predictions
from stage3_detection import COCO_ENV_CLASSES, COCO_CLASS_NAMES


class ModelBundle:
    """Holds every trained/loaded model needed to prepare one sample's features."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None       # ultralytics RTDETR, lazy-loaded
        self.pose_model = None     # ultralytics YOLO-pose, lazy-loaded
        self.appearance_cnn = AppearanceCNN(self.device)
        self.gcn_ae = GCNAutoencoder().to(self.device)
        self.appearance_lstm = AppearanceTemporalLSTM(
            num_nodes=1 + INTENTION.M_vehicles + INTENTION.N_pedestrians + INTENTION.L_traffic_lights,
            hidden_dim=INTENTION.decoder_hidden_dim).to(self.device)
        self.stgcn = STGCNN().to(self.device)
        self.trajectory_ae = TrajectoryLSTMAutoencoder().to(self.device)
        self.skeleton_gcn = SkeletonGCNExtractor().to(self.device)
        self.decoder = IntentionPredictionDecoder().to(self.device)

    def _get_detector(self):
        if self.detector is None:
            from stage3_detection import _load_model
            self.detector = _load_model(DETECTION.model_name)
        return self.detector

    def _get_pose_model(self):
        if self.pose_model is None:
            from ultralytics import YOLO
            self.pose_model = YOLO("yolo11n-pose.pt")  # HRNet substitute, see stage11 docstring
        return self.pose_model


def detect_entities_in_frame(frame_path, model_bundle, conf=DETECTION.confidence_inference_threshold):
    r = model_bundle._get_detector().predict(frame_path, conf=conf, imgsz=DETECTION.imgsz,
                                              classes=COCO_ENV_CLASSES, verbose=False)[0]
    persons, vehicles, lights = [], [], []
    for box in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0])
        d = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        if cls_id == 0:
            persons.append(d)
        elif cls_id == 9:
            lights.append(d)
        else:
            vehicles.append(d)
    return persons, vehicles, lights


def pick_nearby(entities, ref_point, k):
    """k nearest entities to ref_point (Euclidean, bottom-center)."""
    scored = sorted(entities, key=lambda e: np.hypot(
        (e["x1"] + e["x2"]) / 2 - ref_point[0], e["y2"] - ref_point[1]))
    return scored[:k]


@torch.no_grad()
def prepare_sample(sample, video_path, ped_track_history, zebra_polygons, models):
    """sample: one dict from build_crossing_samples(). ped_track_history:
    the observed pedestrian's own boxes across the window, from stage4's
    tracks.json (already filtered to this window by the caller).
    Returns (m_seq, a_seq, r_seq, s_seq, H_aK, C_aK) tensors, batch dim=1,
    ready for the decoder — or None if too few real frames were decodable
    (e.g. a corrupt window) so the caller can skip the sample."""
    device = models.device
    t_start, t_end = sample["window"]
    dense_dir = f"work/intention/dense/{sample['track_id']}_{sample['K2']}_{t_start:.2f}"
    frames = env_graph.extract_dense_window(video_path, t_start, t_end, INTENTION.K_history_frames, dense_dir)
    if len(frames) < 3:
        return None  # need at least 3 frames for Eq. 3-4's acceleration term

    K = len(frames)
    appearance_graphs, motion_graphs = [], []
    ped_boxes_hist = []  # running history for this pedestrian, for motion features
    kp_seq, kp_conf_seq = [], []
    pose_model = models._get_pose_model()

    for i, fr in enumerate(frames):
        persons, vehicles, lights = detect_entities_in_frame(fr["file"], models)
        # observed pedestrian's box at this timestep: nearest ByteTrack
        # observation to this dense frame's timestamp
        ped_box = min(ped_track_history, key=lambda o: abs(o["timestamp_sec"] - fr["timestamp_sec"]))
        ped_box = {"x1": ped_box["x1"], "y1": ped_box["y1"], "x2": ped_box["x2"], "y2": ped_box["y2"]}
        ref_pt = env_graph.bottom_center(ped_box)

        other_peds = [p for p in persons if p is not ped_box]  # crude self-exclusion by ref
        nearby_vehicles = pick_nearby(vehicles, ref_pt, INTENTION.M_vehicles)
        nearby_peds = pick_nearby(other_peds, ref_pt, INTENTION.N_pedestrians)
        nearby_lights = pick_nearby(lights, ref_pt, INTENTION.L_traffic_lights)

        appearance_graphs.append(env_graph.build_appearance_subgraph(
            fr["file"], ped_box, nearby_vehicles, nearby_peds, nearby_lights, models.appearance_cnn))

        ped_boxes_hist.append(ped_box)
        veh_tracklets = [[v] for v in nearby_vehicles]   # single-frame tracklets (simplified: see stage9 note)
        ped_tracklets = [[p] for p in nearby_peds]
        motion_graphs.append(env_graph.build_motion_subgraph(
            ped_boxes_hist[-3:], veh_tracklets, ped_tracklets, zebra_polygons))

        # pose estimation on the pedestrian crop, confidence-gated (stage11)
        x1, y1, x2, y2 = [int(max(v, 0)) for v in (ped_box["x1"], ped_box["y1"], ped_box["x2"], ped_box["y2"])]
        import cv2
        img = cv2.imread(fr["file"])
        crop = img[y1:y2, x1:x2] if img is not None else None
        if crop is not None and crop.size > 0:
            pose_result = pose_model.predict(crop, verbose=False)[0]
            if pose_result.keypoints is not None and len(pose_result.keypoints.xy) > 0:
                kp_seq.append(pose_result.keypoints.xy[0].cpu().numpy())
                kp_conf_seq.append(pose_result.keypoints.conf[0].cpu().numpy()
                                    if pose_result.keypoints.conf is not None else np.zeros(17))
            else:
                kp_seq.append(np.zeros((17, 2))); kp_conf_seq.append(np.zeros(17))
        else:
            kp_seq.append(np.zeros((17, 2))); kp_conf_seq.append(np.zeros(17))

    # --- appearance branch ---
    X_app = torch.tensor(np.stack([g["X"] for g in appearance_graphs]), dtype=torch.float32).to(device)
    A_app = torch.tensor(np.stack([g["A"] for g in appearance_graphs]), dtype=torch.float32).to(device)
    with torch.no_grad():
        Z, _ = models.gcn_ae.encode(X_app, A_app)  # (K, N, 256)
    Z_seq = Z.unsqueeze(0)  # (1, K, N, 256)
    H_aK, C_aK = models.appearance_lstm(Z_seq)
    a_seq = X_app[:, 0, :].unsqueeze(0)  # pedestrian's own (index 0) raw appearance feature per frame

    # --- motion branch (ST-GCNN) ---
    X_mot = torch.tensor(np.stack([g["X"] for g in motion_graphs]), dtype=torch.float32).to(device).unsqueeze(0)
    A_mot = torch.tensor(np.stack([g["A"] for g in motion_graphs]), dtype=torch.float32).to(device).unsqueeze(0)
    m_node_seq = models.stgcn(X_mot, A_mot)         # (1, K, N_motion, 256)
    m_seq = m_node_seq.mean(dim=2)                  # GAP over nodes -> (1, K, 256), Eq. 12's GAP(mt)

    # --- trajectory branch: reuse the trajectory AE's per-step hidden states ---
    traj_xy = np.array([env_graph.bottom_center(b) for b in ped_boxes_hist], dtype=np.float32)
    if len(traj_xy) < K:
        traj_xy = np.pad(traj_xy, ((K - len(traj_xy), 0), (0, 0)), mode="edge")
    traj_t = torch.tensor(traj_xy, dtype=torch.float32).unsqueeze(0).to(device)  # (1,K,2)
    lstm_out, _ = models.trajectory_ae.encoder(traj_t)  # (1,K,trajectory_embed_dim) per-step hidden states
    r_seq = lstm_out

    # --- skeleton branch ---
    kp_xy = np.stack(kp_seq)          # (K,17,2)
    kp_conf = np.stack(kp_conf_seq)   # (K,17)
    feats = skeleton_node_features(kp_xy, kp_conf)
    P = torch.tensor(feats["P"], dtype=torch.float32).unsqueeze(0).to(device)
    M = torch.tensor(feats["M"], dtype=torch.float32).unsqueeze(0).to(device)
    Blen = torch.tensor(feats["B_len"], dtype=torch.float32).unsqueeze(0).to(device)
    Bang = torch.tensor(feats["B_ang"], dtype=torch.float32).unsqueeze(0).to(device)
    skel_embed = models.skeleton_gcn(P, M, Blen, Bang)          # (1, skeleton_embed_dim)
    s_seq = skel_embed.unsqueeze(1).expand(-1, K, -1)            # broadcast over K (single pooled embedding/window)

    return m_seq.detach(), a_seq.detach(), r_seq.detach(), s_seq.detach(), H_aK.detach(), C_aK.detach()


def pretrain_unsupervised(tracks_path, models, sample_frame_paths, epochs_gcn=50, epochs_traj=100):
    """Phase 1. sample_frame_paths: a handful of frames (from stage2's
    sparse manifest is fine here) to build appearance sub-graphs from, for
    GCN-AE pretraining. Uses stage4's tracks for trajectory-AE pretraining."""
    print("Phase 1a: collecting appearance sub-graphs for GCN-AE pretraining...")
    appearance_samples = []
    for fp in sample_frame_paths:
        persons, vehicles, lights = detect_entities_in_frame(fp, models)
        for ped in persons[:3]:  # a few per frame is enough for pretraining diversity
            ref_pt = env_graph.bottom_center(ped)
            g = env_graph.build_appearance_subgraph(
                fp, ped, pick_nearby(vehicles, ref_pt, INTENTION.M_vehicles),
                pick_nearby([p for p in persons if p is not ped], ref_pt, INTENTION.N_pedestrians),
                pick_nearby(lights, ref_pt, INTENTION.L_traffic_lights), models.appearance_cnn)
            appearance_samples.append(g)
    if appearance_samples:
        models.gcn_ae = train_gcn_autoencoder(appearance_samples, epochs=epochs_gcn, device=models.device)
    else:
        print("WARNING: no appearance samples collected — skipping GCN-AE pretraining "
              "(will train from random init in Phase 2, which needs more data to converge).")

    print("Phase 1b: pretraining trajectory LSTM-AE on all tracks...")
    with open(tracks_path) as f:
        tracks = json.load(f)["tracks"]
    trajs = []
    for obs in tracks.values():
        if len(obs) < INTENTION.K_history_frames:
            continue
        pts = np.array([env_graph.bottom_center(o) for o in obs[-INTENTION.K_history_frames:]], dtype=np.float32)
        trajs.append(pts)
    if trajs:
        models.trajectory_ae = train_trajectory_autoencoder(trajs, epochs=epochs_traj, device=models.device)
    else:
        print("WARNING: no trajectories long enough for pretraining.")


@torch.no_grad()
def infer_intention(decoder, models, samples, video_path, tracks, zebra_polygons):
    """Runs the trained decoder over a list of samples (any split), returns
    a list of {"track_id","window","K2","probability"} — stage13's
    expected `intention_predictions.json` format."""
    decoder.eval()
    results = []
    for s in samples:
        ped_hist = tracks.get(s["track_id"], [])
        window_hist = [o for o in ped_hist if s["window"][0] - 1 <= o["timestamp_sec"] <= s["window"][1] + 1]
        if not window_hist:
            continue
        feats = prepare_sample(s, video_path, window_hist, zebra_polygons, models)
        if feats is None:
            continue
        m_seq, a_seq, r_seq, s_seq, H_aK, C_aK = feats
        logit = decoder(m_seq, a_seq, r_seq, s_seq, H_aK, C_aK)
        prob = torch.sigmoid(logit).item()
        results.append({"track_id": s["track_id"], "window": list(s["window"]),
                         "K2": s["K2"], "probability": round(prob, 4)})
    return results


def main():
    ap = argparse.ArgumentParser(description="End-to-end crossing-intention training")
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", default="work/tracks/tracks.json")
    ap.add_argument("--zebra", default="work/zebra/zebra_crossings.json")
    ap.add_argument("--crossing_labels", required=True)
    ap.add_argument("--frames_manifest", default="work/frames_manifest.json",
                     help="stage2's sparse manifest, reused only for GCN-AE pretraining frames")
    ap.add_argument("--out_dir", default="work/intention")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.zebra) as f:
        zebra_polygons = json.load(f)
    with open(args.tracks) as f:
        tracks = json.load(f)["tracks"]

    models = ModelBundle()

    with open(args.frames_manifest) as f:
        manifest = json.load(f)
    pretrain_frames = [fr["file"] for fr in manifest["frames"][:30]]  # a modest subset is enough
    pretrain_unsupervised(args.tracks, models, pretrain_frames)

    print("Phase 2: building labelled samples and preparing features...")
    samples = build_crossing_samples(args.tracks, args.crossing_labels)
    samples = split_samples_by_segment(samples, args.tracks)
    train_samples = [s for s in samples if s["split"] == "train"]

    feature_batches, labels = [], []
    for s in train_samples:
        ped_hist = tracks.get(s["track_id"], [])
        window_hist = [o for o in ped_hist if s["window"][0] - 1 <= o["timestamp_sec"] <= s["window"][1] + 1]
        if not window_hist:
            continue
        feats = prepare_sample(s, args.video, window_hist, zebra_polygons, models)
        if feats is None:
            continue
        feature_batches.append(feats)
        labels.append(torch.tensor([float(s["label"])]))

    print(f"Prepared {len(feature_batches)}/{len(train_samples)} training samples "
          f"(some skipped if their dense window failed to decode).")

    if not feature_batches:
        print("No usable training samples — check --crossing_labels and --video paths.")
        return

    from stage12_intention_decoder import train_intention_decoder
    models.decoder = train_intention_decoder(models.decoder, feature_batches, labels,
                                              device=models.device)

    ckpt_path = os.path.join(args.out_dir, "intention_decoder.pt")
    torch.save(models.decoder.state_dict(), ckpt_path)
    print(f"Saved decoder -> {ckpt_path}")

    # Evaluate on ALL samples so every tracked pedestrian gets a prediction
    eval_samples = samples  # run inference on all samples (train + val + test)
    if eval_samples:
        preds = infer_intention(models.decoder, models, eval_samples, args.video, tracks, zebra_polygons)
        # Also compute metrics on held-out samples if they exist
        held_out = [s for s in samples if s.get("split") in ("val", "test")]
        if held_out:
            y_true = [s["label"] for s in held_out if any(
                p["track_id"] == s["track_id"] and p["window"] == list(s["window"]) for p in preds)]
            y_prob = [p["probability"] for p in preds if any(
                s["track_id"] == p["track_id"] and list(s["window"]) == p["window"] for s in held_out)]
            if y_true and len(y_true) == len(y_prob):
                metrics = evaluate_predictions(y_true, y_prob)
                print("Held-out val/test metrics:", json.dumps(metrics, indent=2))
        pred_path = os.path.join(args.out_dir, "intention_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(preds, f, indent=2)
        print(f"Predictions -> {pred_path}  (feed this + stage7's per_person_summary.json "
              f"into stage13_combined_output.py)")


if __name__ == "__main__":
    main()
