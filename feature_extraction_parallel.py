"""
Parallel feature extraction for graphs (S, M, D)

- Loads:
    data/all_real_graphs.pkl
    data/all_synthetic_graphs.pkl

- Produces:
    data/features_real_parts_*.pkl  (intermediate)
    data/features_synthetic_parts_*.pkl
    data/features_real.pkl          (final: X_real, y_real, meta_real)
    data/features_synthetic.pkl     (final: X_syn, y_syn, meta_syn)
    data/features_combined.pkl      (final: X_all, y_all, meta_all)

Notes:
- This script is CPU-bound. It defaults to 12 worker processes (change NUM_WORKERS).
- It periodically checkpoints partial results (every SAVE_INTERVAL processed items).
- For stability it handles tiny graphs and numerical exceptions.
"""

import os
import pickle
import math
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from itertools import islice

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
from scipy.stats import skew
from tqdm import tqdm


# -----------------------
# Configuration
# -----------------------
NUM_WORKERS = 12  # default worker processes; change if you want to use more/less
SAVE_INTERVAL = 2000  # how many processed graphs between checkpoint saves
SPECTRAL_K = 16  # number of eigenvalues to compute (top-k of normalized Laplacian)


# -----------------------
# Utilities: feature computations
# -----------------------
def safe_spectral_features(G, k=SPECTRAL_K):
    """
    Return vector of length k + 2 (k eigenvalues, mean, variance).
    If graph too small or numerical error, return zeros vector.
    """
    n = G.number_of_nodes()
    if n <= 2:
        return np.zeros(k + 2, dtype=float)

    try:
        A = nx.to_scipy_sparse_array(G, format="csr")
        L = csgraph.laplacian(A, normed=True)
        k_use = min(k, max(1, n - 2))
        vals = eigsh(L, k=k_use, which="SM", return_eigenvectors=False)
        vals = np.sort(vals)
        if len(vals) < k:
            vals = np.pad(vals, (0, k - len(vals)), mode="constant")
        mean_val = float(np.mean(vals))
        var_val = float(np.var(vals))
        out = np.concatenate([vals.astype(float), np.array([mean_val, var_val], dtype=float)])
        return out
    except Exception:
        # fallback: zeros
        return np.zeros(k + 2, dtype=float)


def safe_motif_features(G):
    """
    motifs: triangles, wedges, normalized triangle density, normalized wedge density
    """
    n = G.number_of_nodes()
    if n < 3:
        return np.zeros(4, dtype=float)
    try:
        tri_dict = nx.triangles(G)
        triangles = sum(tri_dict.values()) / 3.0
        degs = [d for _, d in G.degree()]
        wedges = sum(d * (d - 1) / 2.0 for d in degs)
        tri_norm = triangles / max(1.0, (n * (n - 1) * (n - 2) / 6.0))
        wedge_norm = wedges / max(1.0, (n * (n - 1) / 2.0))
        return np.array([float(triangles), float(wedges), float(tri_norm), float(wedge_norm)], dtype=float)
    except Exception:
        return np.zeros(4, dtype=float)


def safe_distribution_features(G):
    """
    distributional: n, m, deg_mean, deg_var, deg_max, deg_skew, clustering_mean, assortativity, density
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    if len(degs) == 0:
        return np.zeros(9, dtype=float)
    try:
        deg_mean = float(np.mean(degs))
        deg_var = float(np.var(degs))
        deg_max = float(np.max(degs))
        deg_skew = float(skew(degs)) if len(degs) > 2 else 0.0
        clust_vals = list(nx.clustering(G).values()) if n > 0 else []
        clust_mean = float(np.mean(clust_vals)) if len(clust_vals) > 0 else 0.0
        try:
            assort = float(nx.degree_assortativity_coefficient(G))
            if not np.isfinite(assort):
                assort = 0.0
        except Exception:
            assort = 0.0
        density = float(nx.density(G))
        return np.array([float(n), float(m), deg_mean, deg_var, deg_max, deg_skew, clust_mean, assort, density], dtype=float)
    except Exception:
        return np.zeros(9, dtype=float)


def compute_feature_vector_for_item(item):
    """
    Worker function. Accepts an item dict with keys:
      - "graph" (NetworkX Graph)
      - metadata keys (domain, dataset, index, etc.)

    Returns a tuple: (feature_vector (np.array), label (int), meta dict)
    For real graphs: label = 0; for synthetic: label = 1.
    For items that are invalid, returns None to skip.
    """
    try:
        G = item.get("graph")
        if G is None:
            return None

        # Determine label (assume synthetic dicts have 'gen_type' key)
        label = 1 if ("gen_type" in item) else 0

        # Compute views
        S = safe_spectral_features(G)
        M = safe_motif_features(G)
        D = safe_distribution_features(G)

        features = np.concatenate([S, M, D]).astype(float)

        # Keep minimal meta for later analysis
        meta = {
            "domain": item.get("domain"),
            "dataset": item.get("dataset") or item.get("source_dataset"),
            "index": item.get("index") or item.get("real_graph_index"),
            "gen_type": item.get("gen_type", None)  # None for real graphs
        }

        return (features, int(label), meta)
    except Exception:
        return None


# -----------------------
# Save helpers
# -----------------------
def save_checkpoint(prefix, X_list, y_list, meta_list, part_id):
    fn = f"data/{prefix}_parts_{part_id}.pkl"
    with open(fn, "wb") as f:
        pickle.dump((X_list, y_list, meta_list), f)


def merge_and_save_all(prefix_parts_pattern, final_path):
    """
    Merge all part files matching prefix_parts_* and save as final_path (pickle).
    """
    import glob
    part_files = sorted(glob.glob(prefix_parts_pattern))
    X_all = []
    y_all = []
    meta_all = []
    for fn in part_files:
        with open(fn, "rb") as f:
            X_part, y_part, meta_part = pickle.load(f)
        X_all.extend(X_part)
        y_all.extend(y_part)
        meta_all.extend(meta_part)
    X_all = np.vstack(X_all) if len(X_all) > 0 else np.zeros((0, 0))
    y_all = np.array(y_all, dtype=int)
    with open(final_path, "wb") as f:
        pickle.dump((X_all, y_all, meta_all), f)
    return final_path


# -----------------------
# Main orchestration
# -----------------------
def process_items_in_parallel(items, prefix, num_workers=NUM_WORKERS, save_interval=SAVE_INTERVAL):
    """
    Process a list of items (dicts) with multiprocessing and checkpoint partial results.
    Returns (final_pickle_path).
    """
    os.makedirs("data", exist_ok=True)
    manager_X = []
    manager_y = []
    manager_meta = []
    part_id = 0
    processed = 0

    worker = compute_feature_vector_for_item

    with Pool(processes=num_workers) as pool:
        # imap_unordered yields results as they complete
        for result in tqdm(pool.imap_unordered(worker, items), total=len(items)):
            if result is None:
                continue
            feat, lbl, meta = result
            manager_X.append(feat)
            manager_y.append(lbl)
            manager_meta.append(meta)
            processed += 1

            # periodic checkpoint
            if processed % save_interval == 0:
                save_checkpoint(prefix, manager_X, manager_y, manager_meta, part_id)
                print(f"\nCheckpoint saved: {prefix}_parts_{part_id}.pkl  (processed {processed}/{len(items)})")
                part_id += 1
                manager_X = []
                manager_y = []
                manager_meta = []

        # save remaining
        if len(manager_X) > 0:
            save_checkpoint(prefix, manager_X, manager_y, manager_meta, part_id)
            print(f"\nFinal checkpoint saved: {prefix}_parts_{part_id}.pkl  (processed total {processed})")
            part_id += 1

    # Merge parts into final file
    final_path = f"data/{prefix}.pkl"
    merged = merge_and_save_all(f"data/{prefix}_parts_*.pkl", final_path)
    print(f"Merged parts saved to: {merged}")
    return final_path


if __name__ == "__main__":
    # Load inputs
    real_path = "data/all_real_graphs.pkl"
    syn_path = "data/all_synthetic_graphs.pkl"
    if (not os.path.exists(real_path)) or (not os.path.exists(syn_path)):
        raise SystemExit("Missing input files. Run Part1 and Part2 first.")

    with open(real_path, "rb") as f:
        real_items = pickle.load(f)

    with open(syn_path, "rb") as f:
        syn_items = pickle.load(f)

    print(f"Real graphs: {len(real_items)}")
    print(f"Synthetic graphs: {len(syn_items)}")
    print("Starting parallel feature extraction")

    # Choose worker count (you can override NUM_WORKERS above or set here)
    available_cpu = cpu_count()
    print(f"Available CPU cores: {available_cpu}. Using NUM_WORKERS = {NUM_WORKERS}")

    # Process real and synthetic separately (separate checkpoints/files)
    t0 = time.time()
    final_real = process_items_in_parallel(real_items, prefix="features_real", num_workers=NUM_WORKERS, save_interval=SAVE_INTERVAL)
    final_syn = process_items_in_parallel(syn_items, prefix="features_synthetic", num_workers=NUM_WORKERS, save_interval=SAVE_INTERVAL)
    t1 = time.time()

    print(f"Feature extraction finished. elapsed seconds: {t1 - t0:.1f}")

    # Merge into combined features
    print("Merging real and synthetic into combined features file...")
    # Load final real and syn
    with open(final_real, "rb") as f:
        X_real, y_real, meta_real = pickle.load(f)
    with open(final_syn, "rb") as f:
        X_syn, y_syn, meta_syn = pickle.load(f)

    X_real = np.vstack(X_real) if len(X_real) > 0 else np.zeros((0, 0))
    X_syn = np.vstack(X_syn) if len(X_syn) > 0 else np.zeros((0, 0))

    X_all = np.vstack([X_real, X_syn])
    y_all = np.concatenate([y_real, y_syn])
    meta_all = meta_real + meta_syn

    combined_path = "data/features_combined.pkl"
    with open("data/features_real.pkl", "wb") as f:
        pickle.dump((X_real, y_real, meta_real), f)
    with open("data/features_synthetic.pkl", "wb") as f:
        pickle.dump((X_syn, y_syn, meta_syn), f)
    with open(combined_path, "wb") as f:
        pickle.dump((X_all, y_all, meta_all), f)

    print("Saved final feature files:")
    print(" - data/features_real.pkl")
    print(" - data/features_synthetic.pkl")
    print(" - data/features_combined.pkl")
