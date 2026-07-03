# Prior-Guided GDN for HAI Anomaly Detection

This repository is adapted from GDN, the implementation of **Graph Neural Network-Based Anomaly Detection in Multivariate Time Series**. The project was modified for the HAI industrial control dataset and explores whether physical prior knowledge can improve graph-based anomaly detection.

## What Changed

The original GDN learns a sensor graph only from trainable node embeddings. In this version, the HAI physical topology is injected into the graph learning process.

Main additions:

- **HAI dataset adaptation**: supports `HAI_train.csv`, `HAI_test.csv`, and HAI sensor lists under `data/<dataset>/`.
- **Physical prior topology**: parses DCS topology JSON files such as `dcs_1001h.json`, `dcs_1002h.json`, and related HAI topology files.
- **Prior-enhanced graph learning**: adds `graph_mode` options:
  - `learned`: original GDN-style learned top-k graph.
  - `prior`: use the physical prior graph directly.
  - `prior_mask`: select learned top-k neighbors only inside prior candidates.
  - `prior_union`: merge the learned graph with HAI physical-prior edges.
- **Unified normalization**: fits MinMax scaling on the training set and applies the same scaler to train/test.
- **Lightweight implementation**: replaces PyG/sklearn/scipy dependencies on the main path with PyTorch/numpy implementations for easier reproduction.

## Best Experiment

The evaluation standard is kept the same as the original project: GDN graph deviation scoring with the existing F1 / precision / recall calculation in `evaluate.py`.

Best run in this workspace:

| Dataset | Graph mode | Prior expand | Top-k | Seed | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hai23.05_end` | `prior_union` | 4 | 15 | 6705 | **0.4180** | 0.6120 | 0.3178 |

Best checkpoint:

```text
pretrained/prior_union_norm_seed6705_best_07-03-09-37-45.pt
```

Training command:

```bash
python main.py \
  -dataset hai23.05_end \
  -device cpu \
  -save_path_pattern prior_union_norm_seed6705_ \
  -slide_stride 1 \
  -slide_win 5 \
  -batch 64 \
  -epoch 20 \
  -random_seed 6705 \
  -decay 0 \
  -dim 64 \
  -out_layer_num 1 \
  -out_layer_inter_dim 128 \
  -val_ratio 0.1 \
  -report best \
  -topk 15 \
  -graph_mode prior_union \
  -prior_expand 4 \
  -prior_min_neighbors 0 \
  -prior_undirected 1 \
  -normalize 1
```

## Observations

The improvement from physical prior knowledge is real but limited. Hard prior constraints (`prior` or `prior_mask`) reduce recall because the HAI physical topology is incomplete for anomaly detection. The best result comes from `prior_union`, which keeps the learned data-driven graph and adds physical-prior edges.

This suggests that, for HAI, the main bottleneck is not only graph construction. GDN's one-step prediction-error assumption is not fully aligned with short, weak, or multi-condition attacks in this dataset. The dataset also contains challenging properties such as constant training columns, train/test distribution shift, and short attack intervals.

## Data Format

Expected dataset layout:

```text
data/
  hai23.05_end/
    list.txt
    HAI_train.csv
    HAI_test.csv
    dcs_1001h.json
    dcs_1002h.json
    dcs_1003h.json
    dcs_1004h.json
    dcs_1010h.json
    dcs_1011h.json
    dcs_1020h.json
```

Notes:

- The first column of CSV files is treated as the index.
- `HAI_test.csv` must include an `attack` column.
- Training labels are treated as normal (`0`) in the current unsupervised setting.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The current implementation only requires PyTorch, numpy, and pandas on the main training/evaluation path.

## Citation

Original GDN paper:

```bibtex
@inproceedings{deng2021graph,
  title={Graph Neural Network-Based Anomaly Detection in Multivariate Time Series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
```
