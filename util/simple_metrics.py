import numpy as np


def _as_int_array(values):
    return np.asarray(values).astype(int).ravel()


def precision_score(y_true, y_pred, zero_division=0):
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = tp + fp
    if denom == 0:
        return zero_division
    return tp / denom


def recall_score(y_true, y_pred, zero_division=0):
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = tp + fn
    if denom == 0:
        return zero_division
    return tp / denom


def f1_score(y_true, y_pred, zero_division=0):
    precision = precision_score(y_true, y_pred, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, zero_division=zero_division)
    if precision + recall == 0:
        return zero_division
    return 2 * precision * recall / (precision + recall)


def mean_squared_error(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def roc_auc_score(y_true, scores):
    y_true = _as_int_array(y_true)
    scores = np.asarray(scores, dtype=float).ravel()
    pos = y_true == 1
    neg = y_true == 0
    pos_n = np.sum(pos)
    neg_n = np.sum(neg)
    if pos_n == 0 or neg_n == 0:
        return 0.0

    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = np.sum(ranks[pos])
    return (pos_rank_sum - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)


def iqr(values):
    values = np.asarray(values, dtype=float)
    return np.percentile(values, 75) - np.percentile(values, 25)


def trim_mean(values, proportiontocut):
    values = np.sort(np.asarray(values, dtype=float).ravel())
    cut = int(len(values) * proportiontocut)
    if cut <= 0:
        return np.mean(values)
    if cut * 2 >= len(values):
        return np.mean(values)
    return np.mean(values[cut:-cut])
