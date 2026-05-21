<h1 align="center">Network Traffic Attribution under Spoofing and Decoy Attacks</h1>

<p align="center">
  A machine-learning pipeline that attributes network flows back to their <em>real</em> source —
  distinguishing genuine attacker traffic from spoofed and decoy flows.<br/>
  Random Forest, XGBoost, and an MLP on 14 hand-crafted flow features.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/sklearn-1.3%2B-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/xgboost-2.0%2B-success.svg" alt="xgboost">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="license">
  <img src="https://img.shields.io/badge/status-research-yellow.svg" alt="status">
</p>

> **Try it instantly — no install needed.** Open `notebooks/colab_demo.ipynb` in Google Colab to clone the repo, install dependencies, load the bundled sample, and run the full pipeline (RF + XGBoost + MLP, metrics, plots) in your browser.

---

## Table of Contents

- [Background](#background)
- [Approach](#approach)
- [Results](#results)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Feature Set](#feature-set)
- [Limitations](#limitations)
- [References](#references)
- [License](#license)

---

## Background

In modern cyberattacks, attackers frequently use **IP spoofing** and **decoy traffic** to hide their true identity. These methods generate misleading network activity that makes it extremely difficult for forensic investigators to identify the *real* source of an attack. BCP38 ingress filtering and hop-count heuristics have known coverage gaps; CAIDA's *Spoofer* survey (2023) reports that roughly 23 % of measured ASes are still spoofable.

A **statistical learning** approach is more robust — it captures the *shape* of attacker traffic rather than the bytes of any single packet. This project trains three detectors (Random Forest, XGBoost, MLP) on a compact set of flow-level features that capture the stability signal: TTL variance, source-IP entropy, TCP window fingerprint, inter-arrival timing, retransmission ratio, byte/packet ratios.

## Approach

```text
           ┌──────────────────────┐
   flows   │   src/preprocess.py  │   load + clean, label-normalize
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │   14 flow features   │   TTL, IP entropy, TCP fingerprint, …
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │     src/train.py     │   Random Forest + XGBoost (5-fold CV)
           │                      │   sklearn MLPClassifier (deep baseline)
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │   src/evaluate.py    │   metrics, confusion matrix, ROC, importances
           └──────────────────────┘
```

The pipeline is **dataset-agnostic**: it works on any CSV with the 14 numeric attribution features (or a subset) plus a binary label column, and equally well on the pre-extracted feature CSVs published with CIC-DDoS2019.

## Results

Metrics on the bundled 10 000-row balanced sample (test split: 2 000 flows):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:------|:--------:|:---------:|:------:|:--:|:-------:|
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MLP (128-64-32) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The synthetic sample is highly separable. Numbers on real captures (CIC-DDoS2019) are expected to sit in the 0.97–0.998 band published by Sharafaldin et al. (2019).

Figures saved by `evaluate.py` to `reports/figures/`:

- `confusion_matrix_rf.png`, `confusion_matrix_xgb.png`
- `roc_comparison.png`
- `feature_importance_rf.png`, `feature_importance_xgb.png`

## Repository Layout

```
Network-Traffic-Attribution/
├── src/                        # production code
│   ├── preprocess.py           # dataset loading + label normalization
│   ├── features.py             # 14 attribution feature definitions
│   ├── generate_sample.py      # synthetic data generator
│   ├── train.py                # RF + XGB training (5-fold CV)
│   ├── predict.py              # score with a pretrained .pkl
│   ├── evaluate.py             # metrics + figures
│   └── __init__.py
├── data/
│   └── sample/                 # bundled 10 K balanced sample
├── models/                     # rf.pkl, xgb.pkl, mlp.pkl (pretrained)
├── notebooks/                  # exploratory + demo notebooks
├── deep_learning/              # MLP description + notebook
├── reports/
│   ├── figures/                # plots saved by evaluate.py
│   ├── metrics.csv             # final metric table
│   ├── report.md / report.docx / report.pdf
│   └── شرح_المشروع.md
├── scripts/build_report.py     # MD → DOCX + PDF builder
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```powershell
cd Network-Traffic-Attribution

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option A — Use the pretrained models *(fastest)*

```powershell
python -m src.predict --model models/rf.pkl
python -m src.predict --model models/xgb.pkl
```

### Option B — Train from scratch

```powershell
python -m src.generate_sample              # writes data/sample/attribution_sample.csv
python -m src.train --data data/sample/attribution_sample.csv
python -m src.evaluate
```

`train.py` does 5-fold CV on Random Forest and XGBoost, saves new `.pkl` files into `models/`, then `evaluate.py` writes confusion-matrix, ROC, and feature-importance PNGs into `reports/figures/`.

---

## Datasets

### Bundled sample (10 000 rows, balanced)

A synthetic attribution dataset ships at [`data/sample/attribution_sample.csv`](data/sample/attribution_sample.csv): 5 000 real-attacker flows + 5 000 spoof/decoy flows, with all 14 attribution features and a binary `label` column. Generated by [`src/generate_sample.py`](src/generate_sample.py) with parameters informed by CIC-DDoS2019 statistics; an 8 % `hard_fraction` injects overlap to prevent trivial separability.

### Full dataset (download separately)

The full evaluation uses **CIC-DDoS2019** released by the Canadian Institute for Cybersecurity. ~12.8 M flow records labelled with the attack type that produced them. Place the downloaded CSVs under `data/raw/` (gitignored) and run:

```powershell
python -m src.train --data data/raw/CIC-DDoS2019/
```

## Feature Set

Each flow is reduced to **14 numeric features** described in [`src/features.py`](src/features.py):

| Feature | Type | Intuition |
|---|---|---|
| `ttl_mean` | float | Real sources use OS-specific TTLs (64, 128, 255) |
| `ttl_variance` | float | Spoofed flows show large TTL jumps |
| `src_ip_entropy` | float | Decoy floods use many random IPs |
| `tcp_window_size_mean` | float | OS-fingerprint signal |
| `tcp_window_size_std` | float | Real sources hold near-constant |
| `tcp_options_hash` | int | Coarse OS fingerprint |
| `packet_size_mean` | float | Real attackers cluster their payloads |
| `packet_size_std` | float | High for chaotic decoy mixes |
| `interarrival_mean` | float | Real sources have predictable cadence |
| `interarrival_variance` | float | Variance is the strongest temporal signal |
| `flow_duration` | float | Flow length |
| `packets_per_second` | float | Burst rate |
| `bytes_per_packet` | float | Payload size |
| `retransmission_ratio` | float | Anomalously high for spoofs |

---

## Limitations

- We use only **per-flow features**. Cross-flow features (route diversity, RTT consistency across multiple flows from one source) would catch more sophisticated attribution evasion.
- We assume a vantage point with access to **per-packet TTL, TCP options, and timing** (a network sensor or a router export). Encrypted protocols don't change visibility here because the features are header/timing-only.
- The bundled sample is **synthetic** — production numbers will be lower than the 1.000 reported here; we expect the 0.97–0.998 band on CIC-DDoS2019.
- An attacker who **carefully shapes a spoof to mimic a real source** is harder to flag.

## References

1. Ferguson, P., & Senie, D. (2000). *Network Ingress Filtering* (BCP38). RFC 2827.
2. Jin, C., Wang, H., & Shin, K. (2003). *Hop-Count Filtering*. ACM CCS.
3. Mirkovic, J., & Reiher, P. (2004). *A Taxonomy of DDoS Defense Mechanisms*.
4. Tang, X., et al. (2014). *DDoS Source Identification Using Random Forests*.
5. Doshi, R., Apthorpe, N., & Feamster, N. (2018). *Machine Learning DDoS Detection for Consumer IoT Devices*.
6. Sharafaldin, I., Lashkari, A. H., Hakak, S., & Ghorbani, A. A. (2019). *CIC-DDoS2019 dataset*. <https://www.unb.ca/cic/datasets/ddos-2019.html>
7. Doriguzzi-Corin, R., et al. (2020). *LUCID: Lightweight Deep Learning for DDoS Detection*.

## License

This project is released under the MIT License — see `LICENSE` for details.
