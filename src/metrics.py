from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Thresholding strategies
# ---------------------------------------------------------------------------

def default_threshold(scores: Sequence[float], smaller_is_member: bool) -> float:
    values = np.array(list(scores), dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def best_f1_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    smaller_is_member: bool,
    n_steps: int = 200,
) -> float:
    """Sweep thresholds and return the one that maximises F1."""
    arr = np.array(list(scores), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return lo
    best_t, best_f1 = lo, 0.0
    for t in np.linspace(lo, hi, n_steps):
        preds = predictions_from_scores(scores, float(t), smaller_is_member)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def predictions_from_scores(scores: Sequence[float], threshold: float, smaller_is_member: bool) -> List[int]:
    if smaller_is_member:
        return [1 if s <= threshold else 0 for s in scores]
    return [1 if s >= threshold else 0 for s in scores]


# ---------------------------------------------------------------------------
# TPR at low FPR
# ---------------------------------------------------------------------------

def tpr_at_fpr(
    labels: Sequence[int],
    scores: Sequence[float],
    smaller_is_member: bool,
    target_fprs: Sequence[float] = (0.01, 0.05, 0.10),
) -> Dict[str, float]:
    """Compute TPR at specified FPR thresholds using the ROC curve."""
    labels_arr = np.array(list(labels))
    scores_arr = np.array(list(scores), dtype=float)
    if smaller_is_member:
        scores_arr = -scores_arr
    if len(set(labels_arr.tolist())) < 2:
        return {f"tpr@fpr={fpr:.2f}": float("nan") for fpr in target_fprs}
    fpr_vals, tpr_vals, _ = roc_curve(labels_arr, scores_arr)
    result: Dict[str, float] = {}
    for target in target_fprs:
        idx = np.searchsorted(fpr_vals, target, side="right") - 1
        idx = max(0, min(idx, len(tpr_vals) - 1))
        result[f"tpr@fpr={target:.2f}"] = float(tpr_vals[idx])
    return result


# ---------------------------------------------------------------------------
# Classification metrics (extended)
# ---------------------------------------------------------------------------

def classification_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    smaller_is_member: bool,
) -> Dict[str, float]:
    labels_list = list(labels)
    scores_list = list(scores)
    if len(set(labels_list)) < 2:
        auc = float("nan")
    else:
        auc_scores = [-s for s in scores_list] if smaller_is_member else scores_list
        auc = float(roc_auc_score(labels_list, auc_scores))

    # Median threshold (original)
    threshold_median = default_threshold(scores_list, smaller_is_member=smaller_is_member)
    preds_median = predictions_from_scores(scores_list, threshold_median, smaller_is_member)
    p_med, r_med, f1_med, _ = precision_recall_fscore_support(
        labels_list, preds_median, average="binary", zero_division=0,
    )

    # Best-F1 threshold
    threshold_f1 = best_f1_threshold(labels_list, scores_list, smaller_is_member)
    preds_f1 = predictions_from_scores(scores_list, threshold_f1, smaller_is_member)
    p_f1, r_f1, f1_f1, _ = precision_recall_fscore_support(
        labels_list, preds_f1, average="binary", zero_division=0,
    )

    # TPR at low FPR
    tpr_metrics = tpr_at_fpr(labels_list, scores_list, smaller_is_member)

    result = {
        "auc": auc,
        "threshold_median": float(threshold_median),
        "precision_median": float(p_med),
        "recall_median": float(r_med),
        "f1_median": float(f1_med),
        "threshold_f1": float(threshold_f1),
        "precision_f1": float(p_f1),
        "recall_f1": float(r_f1),
        "f1_f1": float(f1_f1),
        # backward compat
        "threshold": float(threshold_median),
        "precision": float(p_med),
        "recall": float(r_med),
        "f1": float(f1_med),
    }
    result.update(tpr_metrics)
    return result


# ---------------------------------------------------------------------------
# Coverage & Stability
# ---------------------------------------------------------------------------

def coverage(sample_ids: Sequence[str], labels: Sequence[int], preds: Sequence[int]) -> float:
    true_members = {sid for sid, label in zip(sample_ids, labels) if label == 1}
    detected_members = {sid for sid, pred in zip(sample_ids, preds) if pred == 1}
    if not true_members:
        return 0.0
    return len(true_members & detected_members) / len(true_members)


def coverage_canaries(sample_ids: Sequence[str], sources: Sequence[str], preds: Sequence[int]) -> float:
    canary_ids = {sid for sid, src in zip(sample_ids, sources) if src == "canary"}
    detected = {sid for sid, pred in zip(sample_ids, preds) if pred == 1}
    if not canary_ids:
        return 0.0
    return len(canary_ids & detected) / len(canary_ids)


def stability(run_predictions: List[Sequence[int]], sample_ids: Sequence[str]) -> float:
    detected_sets: List[Set[str]] = []
    for preds in run_predictions:
        detected = {sid for sid, pred in zip(sample_ids, preds) if pred == 1}
        detected_sets.append(detected)

    if len(detected_sets) < 2:
        return 1.0

    pairwise = []
    for i in range(len(detected_sets)):
        for j in range(i + 1, len(detected_sets)):
            a, b = detected_sets[i], detected_sets[j]
            union = a | b
            if not union:
                pairwise.append(1.0)
            else:
                pairwise.append(len(a & b) / len(union))
    return float(np.mean(pairwise)) if pairwise else 1.0


# ---------------------------------------------------------------------------
# Statistical aggregation across runs
# ---------------------------------------------------------------------------

def aggregate_run_metrics(
    run_metrics: List[Dict[str, float]],
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across runs: mean, std, CI."""
    if not run_metrics:
        return {}
    keys = [k for k in run_metrics[0] if isinstance(run_metrics[0][k], (int, float))]
    agg: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.array([m[key] for m in run_metrics if not np.isnan(m.get(key, float("nan")))], dtype=float)
        if len(vals) == 0:
            agg[key] = {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            ci = scipy_stats.t.interval(confidence, df=len(vals) - 1, loc=mean, scale=std / np.sqrt(len(vals)))
            ci_low, ci_high = float(ci[0]), float(ci[1])
        else:
            ci_low, ci_high = mean, mean
        agg[key] = {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high}
    return agg
