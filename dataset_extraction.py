"""
Part 1: Complete Real Graph Dataset Extraction

This script loads:
1. Social TU datasets
2. Bioinformatics TU datasets
3. Citation datasets (Cora, Citeseer, PubMed) via Planetoid,
   and extracts ego-nets to create multiple graphs

All graphs are converted into NetworkX.
All graphs are saved into: data/all_real_graphs.pkl
"""

import os
import pickle
import random
import networkx as nx
from torch_geometric.datasets import TUDataset, Planetoid


# -------------------------------------------------------------
# Convert PyG Data to NetworkX
# -------------------------------------------------------------
def pyg_to_nx(pyg_data):
    edge_index = pyg_data.edge_index.numpy()
    num_nodes = pyg_data.num_nodes

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for u, v in edge_index.T:
        if u != v:
            G.add_edge(int(u), int(v))

    return G


# -------------------------------------------------------------
# Load TU datasets (social + bio)
# -------------------------------------------------------------
def load_tu_category(base_dir, names, domain):
    all_graphs = []

    for name in names:
        path = os.path.join(base_dir, domain)
        dataset = TUDataset(root=path, name=name)

        for idx, data in enumerate(dataset):
            G = pyg_to_nx(data)
            all_graphs.append({
                "graph": G,
                "domain": domain,
                "dataset": name,
                "index": idx
            })

    return all_graphs


# -------------------------------------------------------------
# Extract ego-nets from a single large graph (citation domain)
# -------------------------------------------------------------
def extract_ego_nets(pyg_data, num_samples=300, hop=2):
    """
    Takes a single PyG graph (Cora/Citeseer/PubMed),
    extracts multiple ego nets of given hop.

    Returns list of networkx graphs.
    """
    G = pyg_to_nx(pyg_data)
    nodes = list(G.nodes())

    samples = []

    for _ in range(num_samples):
        center = random.choice(nodes)
        ego = nx.ego_graph(G, center, radius=hop)
        samples.append(ego)

    return samples


def load_citation_graphs(base_dir):
    names = ["Cora", "Citeseer", "PubMed"]
    output = []

    for name in names:
        dataset = Planetoid(root=os.path.join(base_dir, "citation"), name=name)

        # The Planetoid dataset has only one graph
        pyg_graph = dataset[0]

        # Extract multiple subgraphs
        ego_nets = extract_ego_nets(pyg_graph, num_samples=300, hop=2)

        for idx, G in enumerate(ego_nets):
            output.append({
                "graph": G,
                "domain": "citation",
                "dataset": name,
                "index": idx
            })

    return output


# -------------------------------------------------------------
# Unified loader
# -------------------------------------------------------------
def load_all_real_graphs(base_dir):

    social_names = ["IMDB-BINARY", "IMDB-MULTI", "COLLAB", "REDDIT-BINARY"]
    bio_names = ["MUTAG", "ENZYMES", "PROTEINS"]

    social = load_tu_category(base_dir, social_names, "social")
    bio = load_tu_category(base_dir, bio_names, "bio")
    citation = load_citation_graphs(base_dir)

    return social + bio + citation


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":

    base_dir = "data"

    # Create folders
    os.makedirs(os.path.join(base_dir, "social"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "bio"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "citation"), exist_ok=True)

    print("Loading all real graph datasets. This may take several minutes...")

    all_graphs = load_all_real_graphs(base_dir)

    print(f"Total graphs collected: {len(all_graphs)}")

    save_path = os.path.join(base_dir, "all_real_graphs.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_graphs, f)

    print(f"Saved all graphs to: {save_path}")
    print("Part 1 completed successfully.")
