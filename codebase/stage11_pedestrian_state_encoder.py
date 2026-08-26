"""
stage11_pedestrian_state_encoder.py
Section III-E — Pedestrian-State Encoder

Three feature extractors for the OBSERVED pedestrian specifically
(as opposed to stage9/10, which model the surrounding environment):

  1. Appearance feature extractor — same GhostNet CNN used for the
     appearance sub-graph nodes (stage9's cnn_extractor). Reused here.
  2. Trajectory feature extractor — LSTM autoencoder (Eq. 10), trained
     UNSUPERVISED on all pedestrian trajectories from stage4's tracks.json.
     Encoder kept, decoder discarded after convergence.
  3. Skeleton feature extractor — Eq. 11 (bone/joint/motion features) +
     a lightweight GCN-based extractor in the spirit of EfficientGCN.

     >>> ALTITUDE / SUBSTITUTION CAVEAT <<<
     The paper uses HRNet for pose estimation on (comparatively low,
     oblique) surveillance camera footage. Your footage is a near-nadir
     high-altitude drone shot — pedestrians are a handful of pixels, and
     HRNet-quality pose estimation is unlikely to produce usable
     keypoints for most detections at that scale. This module:
       - uses Ultralytics YOLO-pose (COCO-17 keypoints) instead of HRNet,
         since it's already a dependency here and works acceptably down
         to somewhat smaller person crops than HRNet's typical operating
         range — but neither will be reliable at extreme altitude.
       - gates on keypoint confidence: any pedestrian crop whose pose
         estimation confidence falls below `pose_conf_threshold` gets a
         ZERO skeleton feature vector (exactly the paper's own occlusion
         handling: "coordinates of their unrecognized joint points are
         set to 0, so as to enable the subsequent network to disregard
         them"). In practice, expect this branch to contribute little to
         nothing on the highest-altitude passes of your footage, and
         progressively more on any lower/closer passes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INTENTION, ATTRIBUTE


# ----------------------------------------------------------------------
# 1. Appearance feature extractor (GhostNet, shared with stage9)
# ----------------------------------------------------------------------
class AppearanceCNN(nn.Module):
    """Wraps timm's GhostNet, returning pooled (1280,) features per crop —
    matches the paper's stated choice (Table I) and its stated feature size."""

    def __init__(self, device=None):
        super().__init__()
        import timm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = timm.create_model("ghostnet_100", pretrained=True, num_classes=0).to(self.device)
        self.backbone.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @torch.no_grad()
    def __call__(self, bgr_crop):
        """bgr_crop: HxWx3 uint8 numpy array (as read by cv2). Returns a
        (1280,) numpy feature vector — used directly as the callable
        `cnn_extractor` argument expected by stage9.build_appearance_subgraph."""
        import cv2
        img = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype("float32") / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = (t - self.mean) / self.std
        feat = self.backbone(t)
        return feat.squeeze(0).cpu().numpy()


# ----------------------------------------------------------------------
# 2. Trajectory feature extractor — LSTM autoencoder, Eq. 10
# ----------------------------------------------------------------------
class TrajectoryLSTMAutoencoder(nn.Module):
    def __init__(self, hidden_dim=INTENTION.trajectory_embed_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=2, hidden_size=hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, 2)

    def encode(self, tr):
        """tr: (B, K, 2) pixel-coordinate trajectory. Returns e: (B, hidden_dim)."""
        _, (h_n, _) = self.encoder(tr)
        return h_n.squeeze(0)  # (B, hidden_dim)

    def forward(self, tr):
        B, K, _ = tr.shape
        e = self.encode(tr)
        h, c = e, torch.zeros_like(e)
        # decoder starts from a <START> token (zeros), per Fig. 4
        dec_input = torch.zeros(B, 2, device=tr.device)
        outputs = []
        for t in range(K):
            h, c = self.decoder_cell(dec_input, (h, c))
            pred = self.out_fc(h)
            outputs.append(pred)
            dec_input = pred
        tr_recon = torch.stack(outputs, dim=1)  # (B, K, 2)
        return e, tr_recon

    @staticmethod
    def loss(tr_recon, tr):
        return F.mse_loss(tr_recon, tr)


def train_trajectory_autoencoder(all_trajectories, epochs=100, batch_size=256, lr=1e-3, device=None):
    """all_trajectories: list of (K,2) arrays, pixel bottom-center coords
    over time, extracted from stage4's tracks.json. Trained unsupervised
    (Eq. 10); returns the trained model (use .encode() only afterward,
    decoder is discarded as in the paper)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryLSTMAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    data = torch.tensor(np.stack(all_trajectories), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(data)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            opt.zero_grad()
            _, recon = model(batch)
            loss = model.loss(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"[Traj-AE] epoch {epoch+1}/{epochs} loss={total/len(dl):.4f}")
    return model


# ----------------------------------------------------------------------
# 3. Skeleton feature extractor — Eq. 11 + GCN, with confidence gating
# ----------------------------------------------------------------------
COCO17_EDGES = [  # standard COCO-17 skeleton adjacency (used for the GCN)
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
CENTER_JOINT = 0  # nose, as a stand-in "center spine joint" (c in Eq. 11) —
                   # COCO-17 has no true spine joint; nose is the closest stable proxy


def coco17_adjacency():
    A = np.zeros((17, 17), dtype=np.float32)
    for i, j in COCO17_EDGES:
        A[i, j] = A[j, i] = 1.0
    return A


def skeleton_node_features(keypoints_xy, keypoints_conf, pose_conf_threshold=0.3):
    """keypoints_xy: (T, 17, 2), keypoints_conf: (T, 17). Implements Eq. 11:
    joint positions P (relative to center joint), motion velocities M
    (frame-to-frame), bone features B (length + angle to adjacent joint).
    Low-confidence joints are zeroed out (paper's occlusion handling)."""
    T, V, _ = keypoints_xy.shape
    kp = keypoints_xy.copy()
    low_conf = keypoints_conf < pose_conf_threshold
    kp[low_conf] = 0.0

    c = kp[:, CENTER_JOINT:CENTER_JOINT + 1, :]  # (T,1,2)
    P = kp - c  # normalized position features, (T,17,2)

    M = np.zeros_like(kp)
    M[1:] = kp[1:] - kp[:-1]  # motion velocities between adjacent frames

    B_len = np.zeros((T, V), dtype=np.float32)
    B_ang = np.zeros((T, V, 2), dtype=np.float32)
    adj_map = {}
    for i, j in COCO17_EDGES:
        adj_map.setdefault(i, j)
        adj_map.setdefault(j, i)
    for v in range(V):
        adj = adj_map.get(v, v)
        l = kp[:, v, :] - kp[:, adj, :]
        B_len[:, v] = np.linalg.norm(l, axis=-1)
        denom = np.linalg.norm(l, axis=-1, keepdims=True) + 1e-8
        B_ang[:, v, :] = np.arccos(np.clip(l / denom, -1, 1))

    # overall mean pose-detection confidence, used as a gate for whether
    # this branch should be trusted at all for this pedestrian/window
    mean_conf = float(keypoints_conf.mean())
    return {"P": P, "M": M, "B_len": B_len, "B_ang": B_ang, "mean_conf": mean_conf}


class SkeletonGCNExtractor(nn.Module):
    """Three branches (joint positions, motion velocities, bone features)
    -> GCN blocks -> fusion -> temporal pooling -> skeleton embedding.
    Lightweight stand-in for the paper's EfficientGCN-based extractor."""

    def __init__(self, hidden_dim=INTENTION.skeleton_embed_dim):
        super().__init__()
        from stage10_environment_encoder import GraphConvLayer
        self.gcl = GraphConvLayer  # reuse the same Eq.7-style graph conv
        self.pos_gcn1 = GraphConvLayer(2, 64)
        self.pos_gcn2 = GraphConvLayer(64, 128)
        self.mot_gcn1 = GraphConvLayer(2, 64)
        self.mot_gcn2 = GraphConvLayer(64, 128)
        self.bone_gcn1 = GraphConvLayer(3, 64)   # len + 2 angle dims
        self.bone_gcn2 = GraphConvLayer(64, 128)
        self.fuse = nn.Linear(128 * 3, hidden_dim)
        A = torch.tensor(coco17_adjacency())
        self.register_buffer("A", A)

    def forward(self, P, M, B_len, B_ang):
        """P,M: (B,T,17,2). B_len:(B,T,17). B_ang:(B,T,17,2). Returns (B, hidden_dim)."""
        Bn, T, V, _ = P.shape
        A_batch = self.A.unsqueeze(0).expand(Bn * T, V, V)
        A_norm = self.gcl.normalize_adjacency(A_batch)

        def run_branch(feat, gcn1, gcn2):
            x = feat.reshape(Bn * T, V, -1)
            x = gcn1(x, A_norm)
            x = gcn2(x, A_norm)
            return x.reshape(Bn, T, V, -1).mean(dim=(1, 2))  # temporal + joint pooling

        p_out = run_branch(P, self.pos_gcn1, self.pos_gcn2)
        m_out = run_branch(M, self.mot_gcn1, self.mot_gcn2)
        bone_feat = torch.cat([B_len.unsqueeze(-1), B_ang], dim=-1)  # (B,T,17,3)
        b_out = run_branch(bone_feat, self.bone_gcn1, self.bone_gcn2)

        fused = torch.cat([p_out, m_out, b_out], dim=-1)
        return self.fuse(fused)  # (B, hidden_dim)
