"""
Part 2: Synthetic Graph Generation

This script:
1. Loads all real graphs from Part 1
2. For each real graph, generates synthetic graphs using:
       - Erdos-Renyi (ER)
       - Barabasi-Albert (BA)
       - Stochastic Block Model (SBM)
3. Matches size and approximate density
4. Saves everything to: data/all_synthetic_graphs.pkl

All small-graph corner cases are handled safely.
"""

import os
import pickle
import random
import networkx as nx


# -------------------------------------------------------------
# Utility: Compute density of a graph
# -------------------------------------------------------------
def graph_density(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n <= 1:
        return 0.0
    return 2 * m / (n * (n - 1))


# -------------------------------------------------------------
# Generate ER graph safely
# -------------------------------------------------------------
def generate_er_graph(n, p):
    if n < 2:
        return None
    try:
        return nx.erdos_renyi_graph(n, p)
    except:
        return None


# -------------------------------------------------------------
# Generate BA graph safely
# -------------------------------------------------------------
def generate_ba_graph(n, density):
    if n < 3:
        return None

    m = int((density * n) / 2)
    m = max(1, m)
    m = min(m, n - 1)

    try:
        return nx.barabasi_albert_graph(n, m)
    except:
        return None


# -------------------------------------------------------------
# Generate SBM graph safely
# -------------------------------------------------------------
def generate_sbm_graph(n, density):
    if n < 5:
        return None

    num_blocks = random.choice([3, 4])

    base_size = n // num_blocks
    sizes = [base_size] * num_blocks
    sizes[0] += n - sum(sizes)

    if any(s <= 0 for s in sizes):
        return None

    p_in = min(1.0, density * 1.5)
    p_out = min(1.0, density * 0.5)

    probs = [
        [p_in if i == j else p_out for j in range(num_blocks)]
        for i in range(num_blocks)
    ]

    try:
        return nx.stochastic_block_model(sizes, probs)
    except:
        return None


# -------------------------------------------------------------
# Generate synthetic graphs for a real graph (safe version)
# -------------------------------------------------------------
def generate_synthetic_variants(G, K=1):
    n = G.number_of_nodes()
    dens = graph_density(G)

    results = []

    if n < 2:
        return results

    # ER
    for _ in range(K):
        g = generate_er_graph(n, dens)
        if g is not None:
            results.append({"graph": g, "gen_type": "ER"})

    # BA
    for _ in range(K):
        g = generate_ba_graph(n, dens)
        if g is not None:
            results.append({"graph": g, "gen_type": "BA"})

    # SBM
    for _ in range(K):
        g = generate_sbm_graph(n, dens)
        if g is not None:
            results.append({"graph": g, "gen_type": "SBM"})

    return results


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":

    real_path = "data/all_real_graphs.pkl"
    if not os.path.exists(real_path):
        print("Real dataset file not found. Run Part 1 first.")
        exit()

    print("Loading real graphs...")
    with open(real_path, "rb") as f:
        real_graphs = pickle.load(f)

    print(f"Loaded {len(real_graphs)} real graphs.")
    print("Generating synthetic graphs (K=1 per generator)...")

    synthetic_all = []

    for idx, item in enumerate(real_graphs):
        G = item["graph"]
        domain = item["domain"]
        dataset = item["dataset"]

        synthetic_list = generate_synthetic_variants(G, K=1)

        for syn in synthetic_list:
            synthetic_all.append({
                "graph": syn["graph"],
                "domain": domain,
                "source_dataset": dataset,
                "gen_type": syn["gen_type"],
                "real_graph_index": item["index"]
            })

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1} / {len(real_graphs)} real graphs")

    save_path = "data/all_synthetic_graphs.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(synthetic_all, f)

    print(f"Saved synthetic graphs to {save_path}")
    print("Part 2 completed successfully.")
