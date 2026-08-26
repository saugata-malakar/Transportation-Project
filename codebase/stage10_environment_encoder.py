"""
stage10_environment_encoder.py
Section III-D — GCN-Based Environment Encoder

Two encoders, both pure PyTorch (no torch_geometric — the graph
convolution is a direct matmul implementation of Eq. 7, which keeps this
installable on Colab/Kaggle without dependency headaches):

  1. GCNAutoencoder — refines the appearance sub-graph (Eq. 7-8). Two GCN+
     ReLU layers as encoder (1280->512->256), two as decoder (256->512->
     1280). Trained UNSUPERVISED (reconstruction only), then the encoder
     half is kept and the decoder discarded, exactly as the paper does.

  2. STGCNN — builds motion features from the motion sub-graph sequence
     (Eq. 9): one graph convolution (time-varying adjacency per Section
     III-C.2, since here the "graph" is a social/spatial graph of nearby
     agents, not a fixed skeleton) + three temporal convolutions at
     different receptive fields (kernel sizes 3/5/7), summed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INTENTION


# ----------------------------------------------------------------------
# Eq. 7 — graph convolution layer (batched, per-sample adjacency)
# ----------------------------------------------------------------------
class GraphConvLayer(nn.Module):
    """X^(l+1) = ReLU(D_hat^-1/2 A_hat D_hat^-1/2 X^(l) W^(l))
    A: (B, N, N) raw adjacency (no self loops yet). X: (B, N, C_in).
    Returns (B, N, C_out)."""

    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation

    @staticmethod
    def normalize_adjacency(A):
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, N, N)
        A_hat = A + I
        deg = A_hat.sum(dim=2)                      # (B, N)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # (B, N, N)
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, X, A_norm):
        out = A_norm @ self.W(X)
        return F.relu(out) if self.activation else out


# ----------------------------------------------------------------------
# GCN-based autoencoder for the appearance sub-graph, Eq. 7-8
# ----------------------------------------------------------------------
class GCNAutoencoder(nn.Module):
    def __init__(self, in_dim=INTENTION.appearance_feat_dim,
                 hidden_dims=INTENTION.gcn_ae_hidden_dims):
        super().__init__()
        h1, h2 = hidden_dims  # (512, 256)
        self.enc1 = GraphConvLayer(in_dim, h1)
        self.enc2 = GraphConvLayer(h1, h2)
        self.dec1 = GraphConvLayer(h2, h1)
        self.dec2 = GraphConvLayer(h1, in_dim, activation=False)

    def encode(self, X, A):
        A_norm = GraphConvLayer.normalize_adjacency(A)
        h = self.enc1(X, A_norm)
        Z = self.enc2(h, A_norm)   # node embedding, (B, N, 256)
        return Z, A_norm

    def decode(self, Z, A_norm):
        h = self.dec1(Z, A_norm)
        X_tilde = self.dec2(h, A_norm)                 # (B, N, 1280)
        A_tilde = torch.sigmoid(Z @ Z.transpose(1, 2))  # (B, N, N), direct from embedding
        return X_tilde, A_tilde

    def forward(self, X, A):
        Z, A_norm = self.encode(X, A)
        X_tilde, A_tilde = self.decode(Z, A_norm)
        return Z, X_tilde, A_tilde

    @staticmethod
    def reconstruction_loss(A_tilde, A, X_tilde, X, gamma=INTENTION.gcn_ae_gamma):
        """Eq. 8"""
        loss_A = torch.linalg.matrix_norm(A_tilde - A, ord="fro").pow(2).mean()
        loss_X = torch.linalg.matrix_norm(X_tilde - X, ord="fro").pow(2).mean()
        return gamma * loss_A + (1 - gamma) * loss_X


# ----------------------------------------------------------------------
# ST-GCNN for the motion sub-graph sequence, Eq. 9
# ----------------------------------------------------------------------
class STGCNN(nn.Module):
    """Input: X (B, K, N, C_in) node features over K timesteps,
    A (B, K, N, N) that timestep's own adjacency (time-varying social
    graph, per Section III-C.2's Â_m stack). One shared-weight graph
    convolution applied per timestep, followed by three parallel temporal
    convolutions (kernel sizes 3/5/7) across the K axis, summed."""

    def __init__(self, in_dim=13, hidden_dim=INTENTION.stgcn_hidden_dim):
        super().__init__()
        self.gcn = GraphConvLayer(in_dim, hidden_dim)
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, A):
        B, K, N, C = X.shape
        # spatial graph conv, shared weights, per-timestep adjacency
        X_flat = X.reshape(B * K, N, C)
        A_flat = A.reshape(B * K, N, N)
        A_norm = GraphConvLayer.normalize_adjacency(A_flat)
        h = self.gcn(X_flat, A_norm)                     # (B*K, N, hidden)
        h = h.reshape(B, K, N, -1)

        # temporal conv operates over K, per-node: reshape to (B*N, hidden, K)
        hidden_dim = h.shape[-1]
        h_t = h.permute(0, 2, 3, 1).reshape(B * N, hidden_dim, K)  # (B*N, C, K)
        temporal_out = sum(conv(h_t) for conv in self.temporal_convs)  # (B*N, C, K)
        temporal_out = temporal_out.reshape(B, N, hidden_dim, K).permute(0, 3, 1, 2)  # (B,K,N,C)

        out = self.out_proj(temporal_out)  # (B, K, N, hidden)
        return out  # per-timestep, per-node motion embedding; caller does GAP for m_t (Eq. 12)


# ----------------------------------------------------------------------
# Appearance temporal LSTM (Fig. 2 / Section III-D.1, second paragraph):
# "a GCN-based autoencoder is built to refine the appearance features,
# followed by feature association in the temporal dimension through an
# LSTM network." This is what actually produces the (C_a^K, H_a^K) pair
# that seeds the intention decoder — not a single frame's GCN embedding.
# ----------------------------------------------------------------------
class AppearanceTemporalLSTM(nn.Module):
    def __init__(self, num_nodes, node_embed_dim=INTENTION.gcn_ae_hidden_dims[-1],
                 hidden_dim=INTENTION.gcn_ae_hidden_dims[-1]):
        super().__init__()
        self.flatten_dim = num_nodes * node_embed_dim
        self.lstm = nn.LSTM(input_size=self.flatten_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, Z_seq):
        """Z_seq: (B, K, N, node_embed_dim) — the GCN autoencoder's node
        embeddings Z at each of the K timesteps in the window.
        Returns (H_K, C_K), each (B, hidden_dim) — final LSTM state."""
        B, K, N, C = Z_seq.shape
        flat = Z_seq.reshape(B, K, N * C)
        _, (h_n, c_n) = self.lstm(flat)
        return h_n.squeeze(0), c_n.squeeze(0)


# ----------------------------------------------------------------------
# Training entry point for the GCN autoencoder (unsupervised)
# ----------------------------------------------------------------------
def train_gcn_autoencoder(appearance_graph_samples, epochs=50, lr=1e-3, device=None):
    """appearance_graph_samples: list of {"A": (N,N) np, "X": (N,1280) np}
    dicts, from stage9.build_appearance_subgraph(). Trains unsupervised
    (reconstruction only, Eq. 8), returns the trained encoder (decoder
    discarded, per the paper: 'the well-trained encoder is used to refine
    features ... and meanwhile decrease the computational cost')."""
    import numpy as np
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    A_batch = torch.tensor(np.stack([s["A"] for s in appearance_graph_samples]), dtype=torch.float32).to(device)
    X_batch = torch.tensor(np.stack([s["X"] for s in appearance_graph_samples]), dtype=torch.float32).to(device)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        Z, X_tilde, A_tilde = model(X_batch, A_batch)
        loss = model.reconstruction_loss(A_tilde, A_batch, X_tilde, X_batch)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"[GCN-AE] epoch {epoch+1}/{epochs} loss={loss.item():.4f}")
    return model
