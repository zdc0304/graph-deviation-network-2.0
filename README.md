# Topology-Informed Graph Deviation Network for Industrial Anomaly Detection

This repository presents a topology-informed extension of Graph Deviation Network (GDN) for anomaly detection in industrial control systems. The implementation is adapted to the HAI 23.05 dataset and incorporates physical process topology from DCS configuration graphs as prior knowledge for multivariate time-series modeling.

The objective of this project is to study whether domain knowledge from industrial process topology can improve graph-based anomaly detection beyond purely data-driven sensor-relationship learning.

## Method Overview

The original GDN learns sensor dependencies from trainable node embeddings and performs anomaly detection through prediction-error-based graph deviation scoring. This project augments that pipeline with HAI physical-topology information and evaluates several ways of combining prior process knowledge with learned sensor relationships.

## Contributions

- **HAI-oriented data pipeline**: supports `HAI_train.csv`, `HAI_test.csv`, and HAI sensor metadata under `data/<dataset>/`.
- **Physical-topology prior construction**: parses DCS topology JSON files such as `dcs_1001h.json`, `dcs_1002h.json`, and related HAI process-graph files.
- **Prior-enhanced graph learning**: implements multiple graph construction strategies through `graph_mode`:
  - `learned`: original GDN-style learned top-k graph.
  - `prior`: use the physical prior graph directly.
  - `prior_mask`: select learned top-k neighbors only inside prior candidates.
  - `prior_union`: merge the learned graph with HAI physical-prior edges.
- **Consistent preprocessing**: fits MinMax scaling on the training set and applies the same transformation to both train and test data.
- **Dependency-light implementation**: replaces PyG/sklearn/scipy usage on the main path with PyTorch/numpy implementations for easier reproduction.

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

Physical prior knowledge provides a measurable but limited improvement on the HAI setting. Hard prior constraints (`prior` or `prior_mask`) tend to reduce recall, indicating that the available topology is useful but incomplete for anomaly detection. The best result is obtained with `prior_union`, which preserves the data-driven learned graph while adding topology-derived edges.

These results suggest that the main limitation on HAI is not graph construction alone. GDN's one-step prediction-error objective is not fully aligned with short, weak, or multi-condition industrial attacks. The dataset also contains challenging properties such as constant training columns, train/test distribution shift, and short attack intervals.

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
