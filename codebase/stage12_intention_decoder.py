"""
stage12_intention_decoder.py
Section III-F — Intention Prediction Decoder (Eq. 12-18)
Section IV-A — Dataset construction (K1/K2 clip slicing)

The decoder is a single custom LSTM cell whose input at each step t is
Xt = FC(GAP(m_t) (+) (a_t (+) r_t (+) s_t)), aggregating:
  - m_t: ST-GCNN motion embedding at step t (stage10, GAP'd over nodes)
  - a_t: appearance feature at step t (stage11 AppearanceCNN)
  - r_t: trajectory feature (stage11 TrajectoryLSTMAutoencoder, same
         embedding e reused at every t — the paper's r1..rK notation
         suggests a per-step trajectory feature; we take the trajectory
         encoder's per-step hidden state sequence rather than only the
         final embedding, which is what "r1,...,rK" implies)
  - s_t: skeleton embedding at step t (stage11 SkeletonGCNExtractor)
The GCN-autoencoder's final embedding (C_a^K, H_a^K) seeds the LSTM's
initial cell/hidden state (Eq. 13-14, t=0 case).

Dataset (Section IV-A): for each labelled crossing event (pedestrian
track_id, cross_start_time), slice a K1=1s clip ending K2 seconds before
the crossing starts, for K2 in {0,1,2,3}s -> up to 4 positive samples per
crossing pedestrian. Negative samples: K1-second clips from non-crossing
pedestrians, cropped at a random point in their track.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INTENTION


# ----------------------------------------------------------------------
# Eq. 12-18 — custom LSTM decoder cell
# ----------------------------------------------------------------------
class IntentionLSTMCell(nn.Module):
    def __init__(self, input_dim=INTENTION.decoder_input_dim, hidden_dim=INTENTION.decoder_hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wxc = nn.Linear(input_dim, hidden_dim)
        self.Whc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wxi = nn.Linear(input_dim, hidden_dim)
        self.Whi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wxf = nn.Linear(input_dim, hidden_dim)
        self.Whf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wxo = nn.Linear(input_dim, hidden_dim)
        self.Who = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, Xt, H_prev, C_prev):
        C_tilde = torch.tanh(self.Wxc(Xt) + self.Whc(H_prev))          # Eq. 15
        I = torch.sigmoid(self.Wxi(Xt) + self.Whi(H_prev))             # Eq. 16
        Fg = torch.sigmoid(self.Wxf(Xt) + self.Whf(H_prev))            # Eq. 17
        O = torch.sigmoid(self.Wxo(Xt) + self.Who(H_prev))             # Eq. 18
        C = Fg * C_prev + I * C_tilde                                   # Eq. 14 (t>0 case)
        H = O * torch.tanh(C)                                          # Eq. 13 (t>0 case)
        return H, C


class IntentionPredictionDecoder(nn.Module):
    def __init__(self, appearance_dim=INTENTION.appearance_feat_dim,
                 trajectory_dim=INTENTION.trajectory_embed_dim,
                 skeleton_dim=INTENTION.skeleton_embed_dim,
                 motion_dim=INTENTION.stgcn_hidden_dim,
                 gcn_ae_dim=INTENTION.decoder_hidden_dim,
                 decoder_input_dim=INTENTION.decoder_input_dim,
                 hidden_dim=INTENTION.decoder_hidden_dim):
        super().__init__()
        fused_in = motion_dim + appearance_dim + trajectory_dim + skeleton_dim
        self.fc_in = nn.Linear(fused_in, decoder_input_dim)             # Eq. 12's f_FC
        self.cell = IntentionLSTMCell(decoder_input_dim, hidden_dim)
        # project GCN-autoencoder's final (C_a^K, H_a^K) into the LSTM's
        # own hidden size, to seed t=0 (Eq. 13-14, t=0 case)
        self.seed_H = nn.Linear(gcn_ae_dim, hidden_dim)
        self.seed_C = nn.Linear(gcn_ae_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)  # binary crossing intention

    def forward(self, m_seq, a_seq, r_seq, s_seq, H_aK, C_aK):
        """m_seq: (B,K,motion_dim) already GAP'd over nodes (Eq.12's GAP(mt))
        a_seq, r_seq, s_seq: (B,K,dim) each
        H_aK, C_aK: (B, gcn_ae_dim) — the appearance branch's own final
        LSTM hidden/cell state (stage10.AppearanceTemporalLSTM output).
        Per Eq. 13-14 these seed H_0=H_a^K, C_0=C_a^K directly; the
        seed_H/seed_C projections below are only a dimension-matching
        safety net (identity-equivalent when dims already match)."""
        H = torch.tanh(self.seed_H(H_aK))
        C = torch.tanh(self.seed_C(C_aK))

        K = m_seq.shape[1]
        for t in range(K):
            Xt = self.fc_in(torch.cat([m_seq[:, t], a_seq[:, t], r_seq[:, t], s_seq[:, t]], dim=-1))
            H, C = self.cell(Xt, H, C)

        logit = self.classifier(H).squeeze(-1)  # (B,)
        return logit  # apply sigmoid + BCE outside (numerically stabler)


# ----------------------------------------------------------------------
# Section IV-A — dataset construction from labelled crossing events
# ----------------------------------------------------------------------
def build_crossing_samples(tracks_path, crossing_labels_path,
                            K1=INTENTION.K1_clip_duration_sec,
                            K2_offsets=INTENTION.K2_offsets_sec):
    """crossing_labels.json format:
        {"<track_id>": {"crosses": true, "cross_start_time_sec": 184.5}, ...}
        (non-crossing pedestrians: {"crosses": false})

    Returns a list of samples:
        {"track_id":.., "label": 1|0, "K2": <sec>, "window": (t_start,t_end)}
    Positive: one sample per K2 offset for each crossing pedestrian, per
    Section IV-A ("K2 varies from 0 to 3 seconds"). Skips pedestrians whose
    waiting time before crossing is under K1+K2 (paper's own filter).
    Negative: one random K1-second window per non-crossing pedestrian
    whose track is at least K1 seconds long."""
    with open(tracks_path) as f:
        tracks = json.load(f)["tracks"]
    with open(crossing_labels_path) as f:
        labels = json.load(f)

    samples = []
    for tid, obs in tracks.items():
        label_info = labels.get(tid)
        if label_info is None:
            continue
        times = sorted(o["timestamp_sec"] for o in obs)
        track_start, track_end = times[0], times[-1]

        if label_info.get("crosses"):
            cross_t = label_info["cross_start_time_sec"]
            for k2 in K2_offsets:
                t_end = cross_t - k2
                t_start = t_end - K1
                if t_start < track_start:
                    continue  # insufficient waiting time before crossing (IV-A filter)
                samples.append({"track_id": tid, "label": 1, "K2": k2,
                                 "window": (round(t_start, 2), round(t_end, 2))})
        else:
            if track_end - track_start < K1:
                continue  # too short to form a complete K1-second sample
            t_end = track_start + K1 + np.random.rand() * (track_end - track_start - K1)
            t_start = t_end - K1
            samples.append({"track_id": tid, "label": 0, "K2": None,
                             "window": (round(t_start, 2), round(t_end, 2))})

    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = sum(1 for s in samples if s["label"] == 0)
    print(f"Built {len(samples)} samples ({n_pos} positive / {n_neg} negative) "
          f"from {len(tracks)} tracks.")
    return samples


def split_samples_by_segment(samples, tracks_path, test_fraction=0.2, val_fraction=0.1,
                              segment_len_sec=20.0):
    """Same video-segment-level split principle as stage2 — never split by
    shuffling individual samples, to avoid the same pedestrian/scene
    leaking across train/val/test (Section IV-A reports a 70/10/20 split)."""
    max_t = max(s["window"][1] for s in samples)
    n_segments = max(int(max_t // segment_len_sec) + 1, 1)
    n_test = max(int(round(n_segments * test_fraction)), 1)
    n_val = max(int(round(n_segments * val_fraction)), 1)
    test_ids = set(range(n_segments - n_test, n_segments))
    val_ids = set(range(n_segments - n_test - n_val, n_segments - n_test))

    for s in samples:
        seg = int(s["window"][1] // segment_len_sec)
        if seg in test_ids:
            s["split"] = "test"
        elif seg in val_ids:
            s["split"] = "val"
        else:
            s["split"] = "train"
    return samples


# ----------------------------------------------------------------------
# Evaluation metrics — Eq. 19 + PR-AUC
# ----------------------------------------------------------------------
def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    # simple PR-AUC via trapezoidal rule over sorted thresholds
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    total_pos = max(np.sum(y_true == 1), 1)
    recalls = tp_cum / total_pos
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz
    pr_auc = float(trapz_fn(precisions, recalls))

    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "accuracy": round(accuracy, 4), "pr_auc": round(pr_auc, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def train_intention_decoder(decoder, feature_batches, labels, epochs=INTENTION.epochs,
                             lr=INTENTION.lr, device=None):
    """feature_batches: list of (m_seq, a_seq, r_seq, s_seq, H_aK, C_aK)
    tensors already on `device`, one tuple per training sample (see
    stage14_train_intention.py for the full data-loading orchestration
    this assumes upstream). labels: (N,) tensor of 0/1."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    decoder = decoder.to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)

    decoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (m_seq, a_seq, r_seq, s_seq, H_aK, C_aK), y in zip(feature_batches, labels):
            opt.zero_grad()
            logit = decoder(m_seq, a_seq, r_seq, s_seq, H_aK, C_aK)
            loss = F.binary_cross_entropy_with_logits(logit, y.float())
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"[Intention decoder] epoch {epoch+1}/{epochs} "
                  f"loss={total_loss/max(len(feature_batches),1):.4f}")
    return decoder
