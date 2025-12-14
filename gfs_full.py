#!/usr/bin/env python3
"""
gfs_full.py

Graph Fusion System (GFS) full training pipeline.

This script trains a multi-view graph classifier that fuses:
1. Neural graph embeddings (GIN over node degree + clustering features)
2. Spectral handcrafted features
3. Motif-based structural features
4. Distributional graph statistics

Key design goals of this implementation:
- Robust handling of noisy, missing, or inconsistent handcrafted features
- Median / IQR based scaling for heavy-tailed graph statistics
- Aggressive numerical safety to avoid NaN/Inf propagation
- Disk caching for graphs, handcrafted features, and scalers
- Modular architecture enabling ablations and research extensions

Stability-oriented defaults:
- Contrastive loss disabled by default
- Gradient clipping enabled
- Defensive padding and reshaping everywhere
"""

import os
import time
import json
import math
import pickle
import random
from collections import defaultdict

import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

# -------------------------
# Paths and basic config
# -------------------------
PATH_REAL = "data/all_real_graphs.pkl"
PATH_SYN = "data/all_synthetic_graphs.pkl"
PATH_FEATURES_COMBINED = "data/features_combined_cleaned.pkl"  # prefer cleaned; fallback to original
PATH_FEATURES_COMBINED_FALLBACK = "data/features_combined.pkl"
CACHE_PYG = "data/cache_pyg_graphs.pkl"
CACHE_HAND = "data/cache_handcrafted.pkl"
SCALER_META = "data/handcrafted_scaler_meta.pkl"
OUT_DIR = "results/gfs_full"
os.makedirs("data", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------------
# Hyperparameters / options
# -------------------------
CFG = {
    "batch_size": 64,
    "epochs": 20,
    "lr": 5e-4,               # slightly lower LR for stability
    "weight_decay": 1e-5,
    "gin_hidden": 128,
    "gin_layers": 3,
    "proj_dim": 128,          # per-view projector output
    "fusion_dim": 256,        # fusion MLP hidden / fused dim
    "num_workers": 4,
    "contrastive_temp": 0.2,
    "lambda_contrast": 1.0,
    "lambda_ortho": 1e-3,
    "save_every": 5,
    "use_contrast": False,    # disable contrastive by default for first stable runs
    "clip_range": 10.0        # symmetric clip for scaled handcrafted features (Option A)
}

# -------------------------
# Feature names ordering (must match features_combined.pkl)
# -------------------------
FEATURE_NAMES = [
    "eig_0","eig_1","eig_2","eig_3","eig_4","eig_5","eig_6","eig_7",
    "eig_8","eig_9","eig_10","eig_11","eig_12","eig_13","eig_14","eig_15",
    "eig_mean","eig_var",
    "triangles","wedges","tri_norm","wedge_norm",
    "n","m","deg_mean","deg_var","deg_max","deg_skew","clust_mean","assortativity","density"
]
FEATURE_DIM_EXPECTED = len(FEATURE_NAMES)  # 31

# View splits (spectral and distributional come from FEATURES; motifs computed)
SPECTRAL_FEATURES = FEATURE_NAMES[0:18]      # 18 dims
DISTRIBUTIONAL_FEATURES = FEATURE_NAMES[22:] # 9 dims (from index 22 onward)

# Motif view: expanded to 12 dims (computed from graph)
MOTIF_DIM = 12

# -------------------------
# Utility loaders
# -------------------------
def load_graph_lists():
    with open(PATH_REAL, "rb") as f:
        real_list = pickle.load(f)
    with open(PATH_SYN, "rb") as f:
        syn_list = pickle.load(f)
    return real_list, syn_list

def load_features_combined():
    # prefer cleaned file if present
    path = PATH_FEATURES_COMBINED if os.path.exists(PATH_FEATURES_COMBINED) else PATH_FEATURES_COMBINED_FALLBACK
    if not os.path.exists(path):
        raise FileNotFoundError(f"No features file found at {PATH_FEATURES_COMBINED} or fallback {PATH_FEATURES_COMBINED_FALLBACK}")
    with open(path, "rb") as f:
        X, y, meta = pickle.load(f)
    X = np.asarray(X, dtype=float)
    # replace NaN/Inf just in case
    X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    return X, np.asarray(y), list(meta)

# -------------------------
# Lightweight PyG serialization helpers
# -------------------------
def pack_pyg_data(data: Data):
    return {
        "x": data.x.cpu().numpy() if hasattr(data, "x") and data.x is not None else np.zeros((1,2), dtype=np.float32),
        "edge_index": data.edge_index.cpu().numpy() if hasattr(data, "edge_index") and data.edge_index is not None else np.zeros((2,0), dtype=np.int64)
    }

def unpack_pyg_data(packed):
    x = torch.tensor(packed["x"], dtype=torch.float)
    edge_index = torch.tensor(packed["edge_index"], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# -------------------------
# Graph -> PyG Data (lightweight)
# -------------------------
def nx_to_light_pyg(G: nx.Graph):
    n = G.number_of_nodes()
    if n == 0:
        x = np.zeros((1, 2), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        if set(G.nodes()) != set(range(n)):
            mapping = {old: i for i, old in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

        deg = np.array([d for _, d in G.degree()], dtype=np.float32).reshape(-1, 1)
        clust_dict = nx.clustering(G)
        clust = np.array([clust_dict.get(i, 0.0) for i in range(n)], dtype=np.float32).reshape(-1, 1)
        x = np.concatenate([deg, clust], axis=1)  # shape [N, 2]

        edges = np.array(list(G.edges()), dtype=np.int64).T
        if edges.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            u, v = edges
            edge_index = np.vstack([u, v])
            edge_index = np.hstack([edge_index, edge_index[::-1, :]])
    return {"x": x, "edge_index": edge_index}

# -------------------------
# Motif feature computation
# -------------------------
def comb2(x):
    return (x * (x - 1)) / 2.0

def comb3(x):
    return (x * (x - 1) * (x - 2)) / 6.0

def compute_motif_features(G: nx.Graph):
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(MOTIF_DIM, dtype=np.float32)

    tri_dict = nx.triangles(G)
    tri_per_node = np.array([tri_dict.get(i, 0) for i in range(n)], dtype=np.float32)
    triangles_total = float(np.sum(tri_per_node) / 3.0)

    degs = np.array([d for _, d in G.degree()], dtype=np.float32)
    wedges_nodes = np.maximum(0.0, comb2(degs) - tri_per_node)
    wedges_total = float(np.sum(wedges_nodes))

    possible_3 = max(1.0, comb3(n))
    tri_norm = triangles_total / possible_3
    wedge_norm = wedges_total / possible_3

    mean_tri_per_node = float(np.mean(tri_per_node)) if n > 0 else 0.0
    max_tri_per_node = float(np.max(tri_per_node)) if n > 0 else 0.0
    num_deg_ge_3 = int(np.sum(degs >= 3))
    num_leaves = int(np.sum(degs == 1))

    num_components = nx.number_connected_components(G)
    if n > 1:
        comps = list(nx.connected_components(G))
        largest = max(comps, key=len)
        if len(largest) > 1:
            sub = G.subgraph(largest)
            try:
                avg_sp = nx.average_shortest_path_length(sub)
            except Exception:
                avg_sp = 0.0
        else:
            avg_sp = 0.0
    else:
        avg_sp = 0.0

    transitivity = nx.transitivity(G)
    avg_clust = float(np.mean(list(nx.clustering(G).values()))) if n > 0 else 0.0

    feat = np.array([
        triangles_total,
        wedges_total,
        tri_norm,
        wedge_norm,
        mean_tri_per_node,
        max_tri_per_node,
        num_deg_ge_3,
        num_leaves,
        num_components,
        avg_sp,
        transitivity,
        avg_clust
    ], dtype=np.float32)

    if feat.shape[0] != MOTIF_DIM:
        feat = np.pad(feat, (0, max(0, MOTIF_DIM - feat.shape[0])), 'constant')[:MOTIF_DIM]
    return feat

# -------------------------
# Safe flatten/truncate/pad helper
# -------------------------
def safe_flatten_row(raw, expected_len):
    arr = np.asarray(raw)
    flat = arr.flatten()
    if flat.size < expected_len:
        out = np.zeros(expected_len, dtype=np.float32)
        out[:flat.size] = flat
        return out
    else:
        return flat[:expected_len].astype(np.float32)

def ensure_vec(v, expected_len):
    """Return numpy 1D float32 array of length expected_len (pad/truncate)."""
    v = np.asarray(v).flatten() if v is not None else np.array([], dtype=float)
    if v.size >= expected_len:
        return v[:expected_len].astype(np.float32)
    out = np.zeros(expected_len, dtype=np.float32)
    out[:v.size] = v.astype(np.float32)
    return out

# -------------------------
# Scaling helpers (median / IQR)
# -------------------------
def median_iqr_scaler_fit(X):
    # X: numpy array [N, D]
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    # avoid tiny iqr
    iqr[iqr < 1e-6] = 1.0
    return median.astype(np.float32), iqr.astype(np.float32)

def median_iqr_scaler_transform(X, median, iqr, clip=CFG["clip_range"]):
    X = np.asarray(X, dtype=np.float32)
    out = (X - median) / (iqr + 1e-12)
    out = np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip)
    out = np.clip(out, -clip, clip)
    return out.astype(np.float32)

# -------------------------
# Build or load caches (with scaling)
# -------------------------
def build_or_load_cache(force_rebuild=False):
    real_list, syn_list = load_graph_lists()
    graphs_all = [it["graph"] for it in real_list] + [it["graph"] for it in syn_list]
    total_graphs = len(graphs_all)
    print(f"Total graphs (real+synthetic): {total_graphs}")

    X_feat, y_feat, meta_feat = load_features_combined()
    feat_rows = X_feat.shape[0]
    print(f"Feature rows available: {feat_rows}")

    need_rebuild = force_rebuild or (not (os.path.exists(CACHE_PYG) and os.path.exists(CACHE_HAND) and os.path.exists(SCALER_META)))
    if not need_rebuild:
        try:
            with open(CACHE_PYG, "rb") as f:
                cached_pyg = pickle.load(f)
            with open(CACHE_HAND, "rb") as f:
                cached_hand = pickle.load(f)
            with open(SCALER_META, "rb") as f:
                scaler_meta = pickle.load(f)
            if not (isinstance(cached_pyg, list) and isinstance(cached_hand, list) and len(cached_pyg) == total_graphs and len(cached_hand) == total_graphs):
                print("Cache sizes mismatch or invalid. Rebuilding cache.")
                need_rebuild = True
        except Exception as e:
            print("Failed to load caches/scaler:", e)
            need_rebuild = True

    if need_rebuild:
        print("Building caches (this may take a while).")
        cached_pyg = []
        cached_hand = []

        # compute spectral + dist scaler from X_feat rows (only available rows)
        name_to_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        spect_idx = np.array([name_to_idx[n] for n in SPECTRAL_FEATURES], dtype=int)
        dist_idx  = np.array([name_to_idx[n] for n in DISTRIBUTIONAL_FEATURES], dtype=int)

        if feat_rows > 0:
            spect_mat = safe_flatten_row(X_feat[:feat_rows, :][:, spect_idx], len(spect_idx) * feat_rows).reshape(feat_rows, -1)[:feat_rows, :spect_idx.size]
            dist_mat = safe_flatten_row(X_feat[:feat_rows, :][:, dist_idx], len(dist_idx) * feat_rows).reshape(feat_rows, -1)[:feat_rows, :dist_idx.size]
            # ensure shapes
            spect_mat = spect_mat.reshape(feat_rows, len(spect_idx))
            dist_mat = dist_mat.reshape(feat_rows, len(dist_idx))
            spect_med, spect_iqr = median_iqr_scaler_fit(spect_mat)
            dist_med, dist_iqr = median_iqr_scaler_fit(dist_mat)
        else:
            spect_med = np.zeros(len(spect_idx), dtype=np.float32)
            spect_iqr = np.ones(len(spect_idx), dtype=np.float32)
            dist_med = np.zeros(len(dist_idx), dtype=np.float32)
            dist_iqr = np.ones(len(dist_idx), dtype=np.float32)

        # First pass: compute motifs for all graphs (raw) so we can compute motif scaler
        motifs_all = []
        t0 = time.time()
        bad_rows = 0
        for i, G in enumerate(tqdm(graphs_all, desc="Converting graphs (pass 1)")):
            # pack lightweight PyG representation
            packed = nx_to_light_pyg(G)
            cached_pyg.append(packed)

            # compute motif (raw)
            motif = compute_motif_features(G)
            motifs_all.append(motif)

            # keep placeholders for handcrafted now; we'll fill properly after motif scaler known
            cached_hand.append({"spectral": None, "motif": motif, "dist": None})

        # compute motif scaler
        motifs_mat = np.vstack([ensure_vec(m, MOTIF_DIM) for m in motifs_all]).astype(np.float32)
        motif_med, motif_iqr = median_iqr_scaler_fit(motifs_mat)

        # Second pass: fill handcrafted entries with scaled + clipped versions
        bad_rows = 0
        for i in range(total_graphs):
            # spectral & dist from features_combined if available
            if i < feat_rows:
                feat_row = safe_flatten_row(X_feat[i], FEATURE_DIM_EXPECTED)
                spect = feat_row[spect_idx].astype(np.float32)
                dist = feat_row[dist_idx].astype(np.float32)
            else:
                spect = np.zeros(len(spect_idx), dtype=np.float32)
                dist  = np.zeros(len(dist_idx), dtype=np.float32)
                bad_rows += 1
                if bad_rows <= 5:
                    print(f"[INFO] No feature row for graph index {i} -> using zeros for spectral+dist")

            motif = motifs_all[i]

            # apply scaling + clipping
            spect_s = median_iqr_scaler_transform(spect, spect_med, spect_iqr, clip=CFG["clip_range"])
            dist_s = median_iqr_scaler_transform(dist, dist_med, dist_iqr, clip=CFG["clip_range"])
            motif_s = median_iqr_scaler_transform(motif, motif_med, motif_iqr, clip=CFG["clip_range"])

            cached_hand[i] = {"spectral": spect_s.astype(np.float32), "motif": motif_s.astype(np.float32), "dist": dist_s.astype(np.float32)}

        t1 = time.time()
        print(f"Cache build finished in {t1-t0:.1f}s. Saving caches + scaler to disk...")
        with open(CACHE_PYG, "wb") as f:
            pickle.dump(cached_pyg, f)
        with open(CACHE_HAND, "wb") as f:
            pickle.dump(cached_hand, f)
        scaler_meta = {
            "spect_med": spect_med, "spect_iqr": spect_iqr,
            "dist_med": dist_med, "dist_iqr": dist_iqr,
            "motif_med": motif_med, "motif_iqr": motif_iqr,
            "clip_range": CFG["clip_range"]
        }
        with open(SCALER_META, "wb") as f:
            pickle.dump(scaler_meta, f)
        print("Caches + scaler saved:", CACHE_PYG, CACHE_HAND, SCALER_META)
    else:
        print("Loading caches and scaler from disk...")
        with open(CACHE_PYG, "rb") as f:
            cached_pyg = pickle.load(f)
        with open(CACHE_HAND, "rb") as f:
            cached_hand = pickle.load(f)
        with open(SCALER_META, "rb") as f:
            scaler_meta = pickle.load(f)
        print("Caches loaded.")

        # assign scaler meta values for returned context
        spect_med = scaler_meta["spect_med"]
        spect_iqr = scaler_meta["spect_iqr"]
        dist_med = scaler_meta["dist_med"]
        dist_iqr = scaler_meta["dist_iqr"]
        motif_med = scaler_meta["motif_med"]
        motif_iqr = scaler_meta["motif_iqr"]

    # save scaler_meta to return (in case used later)
    scaler_meta = {
        "spect_med": spect_med, "spect_iqr": spect_iqr,
        "dist_med": dist_med, "dist_iqr": dist_iqr,
        "motif_med": motif_med, "motif_iqr": motif_iqr,
        "clip_range": CFG["clip_range"]
    }

    return cached_pyg, cached_hand, X_feat, y_feat, meta_feat, scaler_meta

# -------------------------
# Unpack selected prefix into Data objects and handcrafted list for training
# -------------------------
def prepare_dataset_from_cache(cached_pyg, cached_hand, X_feat, y_feat):
    N_features = X_feat.shape[0]
    total_cached = len(cached_pyg)
    if total_cached < N_features:
        raise RuntimeError(f"Cached graphs ({total_cached}) < feature rows ({N_features}). Rebuild cache or regenerate features.")

    # We'll use only first N_features graphs (labels length)
    data_list = []
    handcrafted = []
    for i in range(N_features):
        data = unpack_pyg_data(cached_pyg[i])
        # defensive: ensure handcrafted entry exists else zeros
        hand_entry = cached_hand[i] if i < len(cached_hand) else {"spectral": np.zeros(len(SPECTRAL_FEATURES), dtype=np.float32),
                                                                   "motif": np.zeros(MOTIF_DIM, dtype=np.float32),
                                                                   "dist": np.zeros(len(DISTRIBUTIONAL_FEATURES), dtype=np.float32)}
        # enforce lengths
        hand_entry["spectral"] = ensure_vec(hand_entry.get("spectral", None), len(SPECTRAL_FEATURES))
        hand_entry["motif"] = ensure_vec(hand_entry.get("motif", None), MOTIF_DIM)
        hand_entry["dist"] = ensure_vec(hand_entry.get("dist", None), len(DISTRIBUTIONAL_FEATURES))
        handcrafted.append(hand_entry)
        data_list.append(data)
    labels = y_feat[:N_features]
    # ensure label numeric type and no NaNs
    labels = np.nan_to_num(labels, nan=0).astype(int)
    return data_list, handcrafted, labels

# -------------------------
# Model components
# -------------------------
class GINEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden=128, num_layers=3, out_dim=128):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden
            nn1 = nn.Sequential(nn.Linear(in_c, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            conv = GINConv(nn1)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden))
        self.project = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        hg = global_mean_pool(h, batch)
        out = self.project(hg)
        return out

class MLPProjector(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class FusionModule(nn.Module):
    def __init__(self, view_dim, fusion_dim):
        """
        view_dim: dimension of each projected view (proj_dim)
        fusion_dim: output fused dimension (CFG['fusion_dim'])
        """
        super().__init__()
        self.view_dim = view_dim
        self.fusion_dim = fusion_dim
        # gate per view
        self.gate = nn.Sequential(
            nn.Linear(view_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1)
        )
        self.project = nn.Sequential(
            nn.Linear(view_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, view_embeddings):
        # view_embeddings: list of tensors [B, D] each
        stacked = torch.stack(view_embeddings, dim=1)  # [B, V, D]
        B, V, D = stacked.shape
        gates = []
        for v in range(V):
            g = self.gate(stacked[:, v, :])  # [B,1]
            gates.append(g.squeeze(1))
        gates = torch.stack(gates, dim=1)  # [B, V]
        weights = F.softmax(gates, dim=1)  # [B, V]
        weighted = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [B, D]
        fused = self.project(weighted)  # [B, fusion_dim]
        return fused, weights

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(8, in_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(8, in_dim // 2), n_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Losses and helpers
# -------------------------
def safe_exp(x):
    # clamp input for stable exp
    return torch.exp(torch.clamp(x, -50, 50))

def nt_xent_loss(z, labels, temperature=0.2):
    z = F.normalize(z, dim=1)
    B = z.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=z.device)
    sim = torch.matmul(z, z.t()) / temperature
    diag = torch.eye(B, device=z.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(diag, -9e15)
    labels = labels.contiguous().view(-1, 1)
    mask_pos = torch.eq(labels, labels.T).float().to(z.device)
    mask_pos = mask_pos - torch.eye(B, device=z.device)
    exp_sim = safe_exp(sim_masked)
    denom = exp_sim.sum(dim=1)
    numer = (exp_sim * mask_pos).sum(dim=1)
    eps = 1e-8
    # avoid log(0)
    frac = (numer + eps) / (denom + eps)
    # clamp inside (eps, 1)
    frac = torch.clamp(frac, min=1e-8, max=1.0)
    loss_per_sample = -torch.log(frac)
    valid = (mask_pos.sum(dim=1) > 0).float()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    loss = (loss_per_sample * valid).sum() / valid.sum()
    return loss

def orthogonality_penalty(projectors):
    mats = []
    for p in projectors:
        for layer in p.net:
            if isinstance(layer, nn.Linear):
                mats.append(layer.weight)
                break
    if len(mats) < 2:
        return torch.tensor(0.0, device=mats[0].device if mats else "cpu")
    total = 0.0
    count = 0
    for i in range(len(mats)):
        for j in range(i+1, len(mats)):
            a = mats[i].view(-1)
            b = mats[j].view(-1)
            if a.device != b.device:
                b = b.to(a.device)
            la = a.numel(); lb = b.numel()
            if la != lb:
                L = max(la, lb)
                a_pad = torch.zeros(L, device=a.device, dtype=a.dtype)
                b_pad = torch.zeros(L, device=a.device, dtype=a.dtype)
                a_pad[:la] = a; b_pad[:lb] = b
                a = a_pad; b = b_pad
            na = a / (a.norm() + 1e-9)
            nb = b / (b.norm() + 1e-9)
            corr = torch.dot(na, nb)
            total += torch.abs(corr)
            count += 1
    return total / max(1, count)

# -------------------------
# Training / Evaluation loops
# -------------------------
def train_loop(model_components, optimiser, dataloader, epoch):
    encoder, proj_neural, proj_spectral, proj_motif, proj_dist, fusion, head = model_components
    encoder.train(); proj_neural.train(); proj_spectral.train(); proj_motif.train(); proj_dist.train(); fusion.train(); head.train()
    running = {"loss": 0.0, "ce": 0.0, "contrast": 0.0, "ortho": 0.0}
    total = 0
    for batch in tqdm(dataloader, desc=f"Train epoch {epoch}", leave=False):
        batch = batch.to(DEVICE)
        labels = batch.y.long().to(DEVICE)
        z_neural = encoder(batch.x, batch.edge_index, batch.batch)
        spect = batch.hand_spect.to(DEVICE)
        motif = batch.hand_motif.to(DEVICE)
        dist = batch.hand_dist.to(DEVICE)

        # defensive reshape
        B = labels.size(0)
        if spect.dim() == 1:
            if spect.numel() == B * len(SPECTRAL_FEATURES):
                spect = spect.view(B, len(SPECTRAL_FEATURES))
            else:
                spect = spect.unsqueeze(0)
        if motif.dim() == 1:
            if motif.numel() == B * MOTIF_DIM:
                motif = motif.view(B, MOTIF_DIM)
            else:
                motif = motif.unsqueeze(0)
        if dist.dim() == 1:
            if dist.numel() == B * len(DISTRIBUTIONAL_FEATURES):
                dist = dist.view(B, len(DISTRIBUTIONAL_FEATURES))
            else:
                dist = dist.unsqueeze(0)

        z_spect = proj_spectral(spect)
        z_motif = proj_motif(motif)
        z_dist = proj_dist(dist)
        z_neural_p = proj_neural(z_neural)
        fused, weights = fusion([z_neural_p, z_spect, z_motif, z_dist])
        logits = head(fused)

        # ensure logits finite
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        ce_loss = F.cross_entropy(logits, labels)
        contrast_loss = nt_xent_loss(fused, labels, temperature=CFG["contrastive_temp"]) if CFG["use_contrast"] else torch.tensor(0.0, device=DEVICE)
        ortho = orthogonality_penalty([proj_neural, proj_spectral, proj_motif, proj_dist])
        loss = ce_loss + CFG["lambda_contrast"] * contrast_loss + CFG["lambda_ortho"] * ortho

        # if loss is NaN, skip this batch (do not step) and log
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARN] NaN/Inf loss encountered; skipping batch update.")
            continue

        optimiser.zero_grad()
        loss.backward()
        # gradient clipping helps stability
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(proj_neural.parameters()) +
                                       list(proj_spectral.parameters()) + list(proj_motif.parameters()) +
                                       list(proj_dist.parameters()) + list(fusion.parameters()) + list(head.parameters()), max_norm=5.0)
        optimiser.step()
        bsize = labels.size(0)
        running["loss"] += loss.item() * bsize
        running["ce"] += ce_loss.item() * bsize
        running["contrast"] += float(contrast_loss.item()) * bsize
        running["ortho"] += float(ortho.item()) * bsize
        total += bsize
    for k in running:
        running[k] /= max(1, total)
    return running

@torch.no_grad()
def eval_loop(model_components, dataloader):
    encoder, proj_neural, proj_spectral, proj_motif, proj_dist, fusion, head = model_components
    encoder.eval(); proj_neural.eval(); proj_spectral.eval(); proj_motif.eval(); proj_dist.eval(); fusion.eval(); head.eval()
    ys = []
    probs = []
    zs = []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = batch.to(DEVICE)
        labels = batch.y.long().to(DEVICE)
        z_neural = encoder(batch.x, batch.edge_index, batch.batch)
        spect = batch.hand_spect.to(DEVICE)
        motif = batch.hand_motif.to(DEVICE)
        dist = batch.hand_dist.to(DEVICE)

        B = labels.size(0)
        if spect.dim() == 1:
            if spect.numel() == B * len(SPECTRAL_FEATURES):
                spect = spect.view(B, len(SPECTRAL_FEATURES))
            else:
                spect = spect.unsqueeze(0)
        if motif.dim() == 1:
            if motif.numel() == B * MOTIF_DIM:
                motif = motif.view(B, MOTIF_DIM)
            else:
                motif = motif.unsqueeze(0)
        if dist.dim() == 1:
            if dist.numel() == B * len(DISTRIBUTIONAL_FEATURES):
                dist = dist.view(B, len(DISTRIBUTIONAL_FEATURES))
            else:
                dist = dist.unsqueeze(0)

        z_spect = proj_spectral(spect)
        z_motif = proj_motif(motif)
        z_dist = proj_dist(dist)
        z_neural_p = proj_neural(z_neural)
        fused, weights = fusion([z_neural_p, z_spect, z_motif, z_dist])
        logits = head(fused)
        prob = F.softmax(logits, dim=1)[:, 1]
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
        ys.append(labels.cpu().numpy())
        probs.append(prob.cpu().numpy())
        zs.append(fused.cpu().numpy())
    ys = np.concatenate(ys) if len(ys) > 0 else np.array([], dtype=int)
    probs = np.concatenate(probs) if len(probs) > 0 else np.array([], dtype=float)
    preds = (probs >= 0.5).astype(int) if probs.size else np.array([], dtype=int)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    metrics = {
        "accuracy": float(accuracy_score(ys, preds)) if ys.size else 0.0,
        "f1": float(f1_score(ys, preds)) if ys.size else 0.0,
        "precision": float(precision_score(ys, preds)) if ys.size else 0.0,
        "recall": float(recall_score(ys, preds)) if ys.size else 0.0,
        "auc": float(roc_auc_score(ys, probs)) if ys.size else 0.0
    }
    zs = np.vstack(zs) if len(zs) > 0 else np.zeros((0, CFG["fusion_dim"]))
    return metrics, zs

# -------------------------
# Dataset wrapper + collate
# -------------------------
class GraphsDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, handcrafted, labels=None):
        assert len(data_list) == len(handcrafted)
        self.data_list = data_list
        self.handcrafted = handcrafted
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        # create a shallow copy to avoid in-place issues
        d_copy = Data(x=d.x, edge_index=d.edge_index)
        h = self.handcrafted[idx]
        # ensure correct lengths and types (defensive)
        spect = ensure_vec(h.get("spectral", None), len(SPECTRAL_FEATURES))
        motif = ensure_vec(h.get("motif", None), MOTIF_DIM)
        dist = ensure_vec(h.get("dist", None), len(DISTRIBUTIONAL_FEATURES))
        d_copy.hand_spect = torch.tensor(spect, dtype=torch.float)
        d_copy.hand_motif = torch.tensor(motif, dtype=torch.float)
        d_copy.hand_dist = torch.tensor(dist, dtype=torch.float)
        if self.labels is not None:
            d_copy.y = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            d_copy.y = torch.tensor(0, dtype=torch.long)
        return d_copy

def collate_graphs(batch):
    # Batch.from_data_list will stack x/edge_index and create batch vector
    batch_out = Batch.from_data_list(batch)
    hand_spect = torch.stack([d.hand_spect for d in batch], dim=0)
    hand_motif = torch.stack([d.hand_motif for d in batch], dim=0)
    hand_dist = torch.stack([d.hand_dist for d in batch], dim=0)
    # double-check dims
    if hand_spect.shape[1] != len(SPECTRAL_FEATURES) or hand_motif.shape[1] != MOTIF_DIM or hand_dist.shape[1] != len(DISTRIBUTIONAL_FEATURES):
        raise RuntimeError(f"handcrafted dims mismatch: spect {hand_spect.shape}, motif {hand_motif.shape}, dist {hand_dist.shape}")
    batch_out.hand_spect = hand_spect
    batch_out.hand_motif = hand_motif
    batch_out.hand_dist = hand_dist
    return batch_out

# -------------------------
# Main entrypoint
# -------------------------
def main():
    print("Device:", DEVICE)
    cached_pyg, cached_hand, X_feat, y_feat, meta_feat, scaler_meta = build_or_load_cache(force_rebuild=False)
    print(f"Cached graphs: {len(cached_pyg)}  Cached handcrafted: {len(cached_hand)}  Feature rows: {X_feat.shape[0]}")

    # Prepare dataset limited by features length
    data_list, handcrafted, labels = prepare_dataset_from_cache(cached_pyg, cached_hand, X_feat, y_feat)
    print(f"Using {len(data_list)} graphs for training (equal to feature rows). Labels shape: {labels.shape}")

    # split
    N = len(data_list)
    idxs = list(range(N))
    random.shuffle(idxs)
    split = int(0.8 * N)
    train_idx = idxs[:split]
    val_idx = idxs[split:]

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    train_hand = [handcrafted[i] for i in train_idx]
    val_hand = [handcrafted[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_dataset = GraphsDataset(train_data, train_hand, train_labels)
    val_dataset = GraphsDataset(val_data, val_hand, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"], collate_fn=collate_graphs)

    # Build model components with correct input dims
    encoder = GINEncoder(in_channels=2, hidden=CFG["gin_hidden"], num_layers=CFG["gin_layers"], out_dim=CFG["proj_dim"]).to(DEVICE)
    proj_neural = MLPProjector(in_dim=CFG["proj_dim"], hidden=CFG["proj_dim"], out_dim=CFG["proj_dim"]).to(DEVICE)
    proj_spectral = MLPProjector(in_dim=len(SPECTRAL_FEATURES), hidden=256, out_dim=CFG["proj_dim"]).to(DEVICE)
    proj_motif = MLPProjector(in_dim=MOTIF_DIM, hidden=256, out_dim=CFG["proj_dim"]).to(DEVICE)
    proj_dist = MLPProjector(in_dim=len(DISTRIBUTIONAL_FEATURES), hidden=128, out_dim=CFG["proj_dim"]).to(DEVICE)

    fusion = FusionModule(view_dim=CFG["proj_dim"], fusion_dim=CFG["fusion_dim"]).to(DEVICE)
    head = ClassifierHead(in_dim=CFG["fusion_dim"], n_classes=2).to(DEVICE)

    params = list(encoder.parameters()) + list(proj_neural.parameters()) + list(proj_spectral.parameters()) + \
             list(proj_motif.parameters()) + list(proj_dist.parameters()) + list(fusion.parameters()) + list(head.parameters())

    optimiser = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    # Sanity forward check (small random batch) to detect shape errors early
    try:
        sample_data = data_list[:2]
        sample_hand = handcrafted[:2]
        sample_batch = [Data(x=sample_data[i].x, edge_index=sample_data[i].edge_index,
                             hand_spect=torch.tensor(ensure_vec(sample_hand[i]["spectral"], len(SPECTRAL_FEATURES)), dtype=torch.float),
                             hand_motif=torch.tensor(ensure_vec(sample_hand[i]["motif"], MOTIF_DIM), dtype=torch.float),
                             hand_dist=torch.tensor(ensure_vec(sample_hand[i]["dist"], len(DISTRIBUTIONAL_FEATURES)), dtype=torch.float)
                             ) for i in range(min(2, len(sample_data)))]
        b = collate_graphs(sample_batch)
        b = b.to(DEVICE)
        b.hand_spect = b.hand_spect.to(DEVICE)
        b.hand_motif = b.hand_motif.to(DEVICE)
        b.hand_dist = b.hand_dist.to(DEVICE)
        enc_out = encoder(b.x, b.edge_index, b.batch)
        _ = proj_spectral(b.hand_spect)
        _ = proj_motif(b.hand_motif)
        _ = proj_dist(b.hand_dist)
        _ = proj_neural(enc_out)
        print("Sanity check forward pass OK.")
    except Exception as e:
        print("Sanity check failed. Error:", e)
        raise

    best_val = 0.0
    history = {"train": [], "val": []}
    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        running = train_loop(
            (encoder, proj_neural, proj_spectral, proj_motif, proj_dist, fusion, head),
            optimiser,
            train_loader,
            epoch
        )
        train_metrics, _ = eval_loop((encoder, proj_neural, proj_spectral, proj_motif, proj_dist, fusion, head), train_loader)
        val_metrics, val_z = eval_loop((encoder, proj_neural, proj_spectral, proj_motif, proj_dist, fusion, head), val_loader)
        t1 = time.time()
        print(f"Epoch {epoch} | time {t1-t0:.1f}s | train_loss {running['loss']:.4f} ce {running['ce']:.4f} contrast {running['contrast']:.4f} ortho {running['ortho']:.6f}")
        print(f"  train acc {train_metrics['accuracy']:.4f} f1 {train_metrics['f1']:.4f} auc {train_metrics['auc']:.4f}")
        print(f"  val   acc {val_metrics['accuracy']:.4f} f1 {val_metrics['f1']:.4f} auc {val_metrics['auc']:.4f}")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        if val_metrics["f1"] > best_val or epoch % CFG["save_every"] == 0:
            best_val = max(best_val, val_metrics["f1"])
            ckpt = {
                "encoder": encoder.state_dict(),
                "proj_neural": proj_neural.state_dict(),
                "proj_spectral": proj_spectral.state_dict(),
                "proj_motif": proj_motif.state_dict(),
                "proj_dist": proj_dist.state_dict(),
                "fusion": fusion.state_dict(),
                "head": head.state_dict(),
                "cfg": CFG,
                "epoch": epoch
            }
            ckpt_path = os.path.join(OUT_DIR, f"gfs_checkpoint_epoch{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(" Saved checkpoint:", ckpt_path)

    with open(os.path.join(OUT_DIR, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Training finished. Results saved in", OUT_DIR)

if __name__ == "__main__":
    main()
