# Graph Fusion System (GFS)
A full-scale multi-view architecture for **real vs synthetic graph detection**.  
This project builds a complete pipeline from dataset construction, handcrafted feature extraction, PyTorch Geometric processing, to a gated fusion model combining neural and statistical graph representations.

---

## Overview
Modern synthetic graph generators increasingly mimic real structural patterns, making authenticity verification challenging.  
GFS provides a scalable solution by integrating four complementary structural views:

1. **Neural View** – GIN-based message passing using local node features  
2. **Spectral View** – Laplacian eigenvalues and global smoothness descriptors  
3. **Motif View** – Higher-order connectivity patterns  
4. **Distributional View** – Degree statistics, clustering behaviour, assortativity, and density  

These views are fused using a **gated attention module**, producing a unified representation for classification.

---

## Key Features
- End-to-end GNN + handcrafted feature fusion  
- Median–IQR scaling for numerical stability  
- Robust caching for 50k+ graphs  
- PyTorch Geometric–based neural encoder  
- Stable gated fusion for multi-view learning  
- Clean and repaired feature matrix with NaN/Inf handling  

---

## Repository Structure
```

GFS/
│── data/                     # Real graphs, synthetic graphs, extracted features
│── models/                   # Model components (GIN, projectors, fusion)
│── results/                  # Checkpoints, logs, plots
│── utils/                    # Helper scripts (feature extraction, caching)
│── gfs_full.py               # Full training pipeline
│── requirements.txt          # Dependencies
│── README.md

````

---

## Installation
```bash
git clone https://github.com/nikunj-indoriya/GFS
cd GFS
pip install -r requirements.txt
````

---

## Running Training

The complete training pipeline is handled by:

```bash
python gfs_full.py
```

This script:

* Loads or builds caches
* Scales and clips handcrafted features
* Trains the GIN encoder
* Projects all views to a unified space
* Performs gated fusion
* Saves checkpoints every 5 epochs

---

## Final Performance (20 Epoch Run)

```
Train F1 ≈ 0.9986  
Val   F1 ≈ 0.9976  
Val   AUC ≈ 0.9999  
```

Pure cross-entropy training (contrastive disabled) provides the most stable results.

---

## Citation

If you use this repository, please cite:

```
@project{gfs2025,
  title={Graph Fusion System for Synthetic vs Real Graph Detection},
  author={Indoriya, Nikunj and Dandhare, Aditya and Fegade, Devesh},
  year={2025},
  institution={IISER Bhopal}
}
```

---

## Contact

For inquiries or collaboration:

**Nikunj Indoriya**
Email: [nikunjindoriya@gmail.com](mailto:nikunjindoriya@gmail.com)
GitHub: [https://github.com/nikunj-indoriya](https://github.com/nikunj-indoriya)
