from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def default_threshold(scores: Sequence[float], smaller_is_member: bool) -> float:
    values = np.array(list(scores), dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def predictions_from_scores(scores: Sequence[float], threshold: float, smaller_is_member: bool) -> List[int]:
    if smaller_is_member:
        return [1 if s <= threshold else 0 for s in scores]
    return [1 if s >= threshold else 0 for s in scores]


def classification_metrics(labels: Sequence[int], scores: Sequence[float], smaller_is_member: bool) -> Dict[str, float]:
    labels = list(labels)
    scores = list(scores)
    if len(set(labels)) < 2:
        auc = float("nan")
    else:
        auc_scores = [-s for s in scores] if smaller_is_member else scores
        auc = float(roc_auc_score(labels, auc_scores))

    threshold = default_threshold(scores, smaller_is_member=smaller_is_member)
    preds = predictions_from_scores(scores, threshold, smaller_is_member=smaller_is_member)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "auc": auc,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


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


