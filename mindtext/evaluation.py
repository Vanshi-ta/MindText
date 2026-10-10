"""Metrics, uncertainty estimates, threshold selection, and trivial baselines.

Binary tasks always use an explicit positive label. All divisions by zero are
reported as 0.0 (``zero_division=0``) and said so in the docs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .errors import ModelError


def _to_str(a) -> np.ndarray:
    return np.asarray(a).astype(str)


def compute_metrics(
    y_true,
    y_pred,
    labels: Sequence[str],
    y_score=None,
    positive_label: str | None = None,
) -> dict:
    y_true, y_pred = _to_str(y_true), _to_str(y_pred)
    labels = [str(l) for l in labels]
    if len(y_true) != len(y_pred):
        raise ModelError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) differ in length.")
    if len(y_true) == 0:
        raise ModelError("Cannot compute metrics on an empty set.")
    unknown = sorted((set(y_true) | set(y_pred)) - set(labels))
    if unknown:
        raise ModelError(f"Labels {unknown} are not in the label set {labels}.")

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out: dict = {
        "n": int(len(y_true)),
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f)),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, l in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
    if len(labels) == 2:
        if positive_label is None:
            raise ModelError("Binary evaluation requires positive_label.")
        pos = str(positive_label)
        if pos not in labels:
            raise ModelError(f"positive_label '{pos}' not in {labels}.")
        pi = labels.index(pos)
        ni = 1 - pi
        tp, fn, fp, tn = int(cm[pi, pi]), int(cm[pi, ni]), int(cm[ni, pi]), int(cm[ni, ni])
        out["binary"] = {
            "positive_label": pos,
            "precision": float(p[pi]),
            "recall": float(r[pi]),
            "f1": float(f[pi]),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "positive_rate_true": float((tp + fn) / len(y_true)),
            "positive_rate_predicted": float((tp + fp) / len(y_true)),
            "roc_auc": None,
            "pr_auc": None,
        }
        if y_score is not None:
            yb = (y_true == pos).astype(int)
            score = np.asarray(y_score, dtype=float)
            if 0 < yb.sum() < len(yb) and np.isfinite(score).all():
                out["binary"]["roc_auc"] = float(roc_auc_score(yb, score))
                out["binary"]["pr_auc"] = float(average_precision_score(yb, score))
    return out


def metric_value(metrics: dict, name: str) -> float:
    """Resolve a selection metric name from a ``compute_metrics`` result."""
    if name == "f1_positive":
        if "binary" not in metrics:
            raise ModelError("'f1_positive' is only defined for binary tasks.")
        return metrics["binary"]["f1"]
    v = metrics.get(name)
    if v is None:
        raise ModelError(f"Metric '{name}' unavailable.")
    return float(v)


# --------------------------------------------------------------------------
# Bootstrap confidence intervals (resampling evaluation records)
# --------------------------------------------------------------------------
def _cm_stats(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    prec = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    rec = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return prec, rec, f1


def bootstrap_ci(
    y_true,
    y_pred,
    labels: Sequence[str],
    positive_label: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CIs for accuracy, macro-F1 (and positive-class
    precision/recall/F1 for binary tasks). Deterministic for a given seed.
    With tiny test sets the intervals are very wide - that is the point."""
    if n_boot <= 0:
        return {}
    labels = [str(l) for l in labels]
    idx = {l: i for i, l in enumerate(labels)}
    t = np.array([idx[v] for v in _to_str(y_true)])
    p = np.array([idx[v] for v in _to_str(y_pred)])
    k, n = len(labels), len(t)
    codes = t * k + p
    rng = np.random.default_rng(seed)
    acc, macro, pos_p, pos_r, pos_f = [], [], [], [], []
    pi = labels.index(str(positive_label)) if (positive_label is not None and k == 2) else None
    for _ in range(n_boot):
        sample = codes[rng.integers(0, n, n)]
        cm = np.bincount(sample, minlength=k * k).reshape(k, k)
        prec, rec, f1 = _cm_stats(cm)
        acc.append(np.trace(cm) / n)
        macro.append(f1.mean())
        if pi is not None:
            pos_p.append(prec[pi]); pos_r.append(rec[pi]); pos_f.append(f1[pi])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def ci(a):
        return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]

    out = {"n_boot": int(n_boot), "alpha": alpha, "accuracy": ci(acc), "macro_f1": ci(macro)}
    if pi is not None:
        out.update({"precision_positive": ci(pos_p), "recall_positive": ci(pos_r), "f1_positive": ci(pos_f)})
    return out


# --------------------------------------------------------------------------
# Thresholds and trivial baselines
# --------------------------------------------------------------------------
def _threshold_metric(y_bin: np.ndarray, pred_bin: np.ndarray, metric: str) -> float:
    tp = float(((y_bin == 1) & (pred_bin == 1)).sum())
    fp = float(((y_bin == 0) & (pred_bin == 1)).sum())
    fn = float(((y_bin == 1) & (pred_bin == 0)).sum())
    tn = float(((y_bin == 0) & (pred_bin == 0)).sum())

    def f1(tp_, fp_, fn_):
        d = 2 * tp_ + fp_ + fn_
        return 2 * tp_ / d if d else 0.0

    if metric == "f1_positive":
        return f1(tp, fp, fn)
    if metric == "f1_macro":
        return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2
    if metric == "balanced_accuracy":
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        return (tpr + tnr) / 2
    raise ModelError(f"Unknown threshold metric '{metric}'.")


def select_threshold(
    y_true,
    scores,
    positive_label: str,
    metric: str = "f1_positive",
    default: float = 0.5,
    max_candidates: int = 101,
) -> tuple[float, float]:
    """Choose the score threshold maximising ``metric`` on the data given
    (callers must pass VALIDATION data only). Candidates are score quantiles.
    Ties go to the candidate closest to ``default`` so that a flat curve does
    not silently produce an "everything positive" rule."""
    y = (_to_str(y_true) == str(positive_label)).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        raise ModelError("Threshold selection needs both classes in the selection data.")
    uniq = np.unique(s)
    if len(uniq) < 2:  # constant scores (e.g. majority baseline): no informative threshold exists
        return float(default), float(_threshold_metric(y, (s >= default).astype(int), metric))
    if len(uniq) > max_candidates:
        uniq = np.unique(np.quantile(s, np.linspace(0, 1, max_candidates)))
    cands = np.unique(np.append(uniq, default))
    vals = np.array([_threshold_metric(y, (s >= c).astype(int), metric) for c in cands])
    best = vals.max()
    tied = cands[np.isclose(vals, best)]
    chosen = float(tied[np.argmin(np.abs(tied - default))])
    return chosen, float(best)


def trivial_baselines(y_true, labels: Sequence[str], positive_label: str | None = None) -> dict:
    """Context for interpreting scores: majority-class and (binary) all-positive."""
    y = _to_str(y_true)
    labels = [str(l) for l in labels]
    counts = {l: int((y == l).sum()) for l in labels}
    majority = max(counts, key=counts.get)
    out = {"majority_class": majority, "majority": compute_metrics(y, np.full(len(y), majority), labels, positive_label=positive_label)}
    out["majority"] = {k: out["majority"][k] for k in ("accuracy", "macro_f1", "weighted_f1")}
    if positive_label is not None and len(labels) == 2:
        pos = str(positive_label)
        allpos = compute_metrics(y, np.full(len(y), pos), labels, positive_label=pos)
        out["all_positive"] = {
            "accuracy": allpos["accuracy"],
            "precision": allpos["binary"]["precision"],
            "recall": allpos["binary"]["recall"],
            "f1": allpos["binary"]["f1"],
        }
    return out
