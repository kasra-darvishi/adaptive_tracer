"""
Apache-style OOD metrics: balance normal vs OOD counts, tune F1 on validation,
report AUROC / AUPR / F1 on test (same spirit as functions.ood_detection_ngram).
"""
from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def balanced_binary_scores_labels(
    scores_id: np.ndarray, scores_ood: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match Apache n-gram protocol: equal normal vs OOD count (min length)."""
    m = min(len(scores_id), len(scores_ood))
    if m == 0:
        return np.array([]), np.array([])
    scores = np.concatenate([scores_id[:m], scores_ood[:m]])
    y = np.concatenate([np.zeros(m, dtype=np.int64), np.ones(m, dtype=np.int64)])
    return scores, y


def tune_threshold_max_f1(
    scores: np.ndarray, y_true: np.ndarray, n_grid: int = 100
) -> tuple[float, float]:
    """Grid-search threshold maximizing F1; higher score => positive (anomaly)."""
    if len(scores) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), 0.0
    smin, smax = float(np.min(scores)), float(np.max(scores))
    if smin == smax or not np.isfinite(smin):
        t = smin if np.isfinite(smin) else 0.0
        y_pred = (scores > t).astype(np.int64)
        return t, float(f1_score(y_true, y_pred, zero_division=0))

    thresholds = np.linspace(smin, smax, n_grid)
    best_t = float(thresholds[0])
    best_f1 = -1.0
    best_precision = -1.0
    for t in thresholds:
        y_pred = (scores > t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        if (
            f1 > best_f1
            or (
                np.isclose(f1, best_f1)
                and (
                    precision > best_precision
                    or (np.isclose(precision, best_precision) and t > best_t)
                )
            )
        ):
            best_f1 = f1
            best_precision = precision
            best_t = float(t)
    return best_t, best_f1


def metrics_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (scores > threshold).astype(np.int64)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def rank_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AUROC and AUPR (requires both classes)."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "aupr": float(average_precision_score(y_true, scores)),
    }
