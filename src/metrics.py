"""评估指标：含抽样加权 AUC/AUPRC、阈值优化与校准。

NHANES 是复杂抽样，必须报告加权结果；加权 AUC 用「权重即病例权重」的
加权 ROC 曲线（按分数排序后累加权重）实现。
"""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score, roc_curve)


def weighted_auc(y, score, w):
    """抽样加权 AUC：按分数降序累加权重构造加权 ROC，再梯形积分。"""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, w = y[order], w[order]

    P, N = w[y == 1].sum(), w[y == 0].sum()
    if P <= 0 or N <= 0:
        return float("nan")
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * (1 - y))
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    return float(np.trapezoid(tpr, fpr))


def weighted_auprc(y, score, w):
    """加权 PR 曲线下面积（阶梯积分）。"""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, w = y[order], w[order]
    P = w[y == 1].sum()
    if P <= 0:
        return float("nan")
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * (1 - y))
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / P
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    return float(np.trapezoid(prec, rec))


def expected_calibration_error(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def best_youden_threshold(y, score, n_points=100):
    """论文1 的做法：在 100 个均匀采样阈值上最大化 Youden 指数。"""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    lo, hi = np.nanmin(score), np.nanmax(score)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5
    grid = np.linspace(lo, hi, n_points)
    best_t, best_j = 0.5, -np.inf
    for t in grid:
        pred = (score >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def binary_metrics(y, score, w=None, threshold=None):
    """完整二分类指标。threshold=None 时按 Youden 指数自动选阈值。"""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if w is None:
        w = np.ones_like(y)

    res = {}
    res["n"] = int(len(y))
    res["prevalence"] = float(y.mean())
    res["weighted_prevalence"] = float(np.average(y, weights=w))
    try:
        res["auc"] = float(roc_auc_score(y, score))
    except ValueError:
        res["auc"] = float("nan")
    res["auc_weighted"] = weighted_auc(y, score, w)
    try:
        res["auprc"] = float(average_precision_score(y, score))
    except ValueError:
        res["auprc"] = float("nan")
    res["auprc_weighted"] = weighted_auprc(y, score, w)

    t = best_youden_threshold(y, score) if threshold is None else threshold
    res["threshold"] = float(t)
    pred = (score >= t).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    res.update(tp=tp, fn=fn, tn=tn, fp=fp)
    res["sensitivity"] = tp / (tp + fn) if (tp + fn) else float("nan")
    res["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
    res["ppv"] = tp / (tp + fp) if (tp + fp) else float("nan")
    res["npv"] = tn / (tn + fn) if (tn + fn) else float("nan")
    res["f1"] = float(f1_score(y, pred, zero_division=0))
    res["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1)))
    res["ece"] = expected_calibration_error(y, score)
    return res


def youden_j(y, score, threshold):
    y = np.asarray(y)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens + spec - 1


def roc_points(y, score):
    fpr, tpr, thr = roc_curve(y, score)
    return fpr, tpr, thr
