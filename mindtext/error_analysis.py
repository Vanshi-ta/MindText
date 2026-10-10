"""Error analysis and (limited) model inspection.

Privacy: excerpts are redacted (URLs, e-mails, @/u/ handles, long digit runs)
and truncated; ``include_text=False`` omits text entirely. Nothing here infers
personal attributes; records are referred to by ``record_id`` only.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

CAVEAT = (
    "Term weights / feature importances describe associations the model learned on this dataset. "
    "They are not causal, do not explain why any individual feels or behaves a certain way, and "
    "must not be read as evidence of a clinical condition."
)

_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HANDLE = re.compile(r"(?<!\w)(?:/?u/|@)[A-Za-z0-9_-]{2,}")
_DIGITS = re.compile(r"\d[\d\s().-]{5,}\d")


def redact(text: str) -> str:
    text = _URL.sub("[URL]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _HANDLE.sub("[USER]", text)
    text = _DIGITS.sub("[NUM]", text)
    return text


def excerpt(text: str, max_chars: int = 160) -> str:
    t = " ".join(redact(str(text)).split())
    return t if len(t) <= max_chars else t[: max(max_chars - 3, 1)].rstrip() + "..."


def _item(row: pd.Series, texts: dict[str, str] | None, include_text: bool, max_chars: int) -> dict[str, Any]:
    item: dict[str, Any] = {
        "record_id": row["record_id"],
        "actual": row["y_true"],
        "predicted": row["y_pred"],
        "score": None if pd.isna(row["score"]) else float(row["score"]),
        "score_kind": row.get("score_kind"),
    }
    if include_text and texts is not None:
        item["excerpt"] = excerpt(texts.get(row["record_id"], ""), max_chars)
    return item


def collect_errors(
    preds: pd.DataFrame,
    positive_label: str | None = None,
    texts: dict[str, str] | None = None,
    n_examples: int = 10,
    include_text: bool = True,
    excerpt_chars: int = 160,
    n_confusions: int = 5,
) -> dict[str, Any]:
    """``preds`` needs columns record_id, y_true, y_pred, score (and optionally score_kind).

    Binary: most-confident false positives / false negatives (a false positive is
    ranked by highest positive-class score, a false negative by lowest).
    Multiclass: the most frequent (actual -> predicted) confusions with their
    most-confident examples. "Score" is a probability only when score_kind == 'proba'."""
    wrong = preds[preds["y_true"] != preds["y_pred"]]
    out: dict[str, Any] = {"n_records": int(len(preds)), "n_errors": int(len(wrong))}
    if positive_label is not None and preds["y_true"].nunique() <= 2:
        pos = str(positive_label)
        fp = wrong[(wrong["y_pred"] == pos)].sort_values("score", ascending=False, kind="mergesort")
        fn = wrong[(wrong["y_true"] == pos)].sort_values("score", ascending=True, kind="mergesort")
        out["positive_label"] = pos
        out["n_false_positives"], out["n_false_negatives"] = int(len(fp)), int(len(fn))
        out["false_positives"] = [_item(r, texts, include_text, excerpt_chars) for _, r in fp.head(n_examples).iterrows()]
        out["false_negatives"] = [_item(r, texts, include_text, excerpt_chars) for _, r in fn.head(n_examples).iterrows()]
        return out
    pairs = wrong.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False, kind="mergesort").head(n_confusions)
    confusions = []
    for (a, b), n in pairs.items():
        sub = wrong[(wrong["y_true"] == a) & (wrong["y_pred"] == b)].sort_values("score", ascending=False, kind="mergesort")
        confusions.append(
            {
                "actual": a,
                "predicted": b,
                "count": int(n),
                "examples": [_item(r, texts, include_text, excerpt_chars) for _, r in sub.head(max(1, n_examples // 2)).iterrows()],
            }
        )
    out["top_confusions"] = confusions
    return out


def top_features(pipe, n: int = 15, feature_names: list[str] | None = None) -> dict[str, Any]:
    """Highest-weighted terms/features per class for linear models, Naive Bayes
    and tree ensembles. Returns {} when the model type is unsupported."""
    clf = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else None
    if clf is None:
        return {}
    if "tfidf" in pipe.named_steps:
        names = np.asarray(pipe.named_steps["tfidf"].get_feature_names_out(), dtype=object)
    elif feature_names is not None:
        names = np.asarray(feature_names, dtype=object)
    else:
        return {}
    classes = [str(c) for c in clf.classes_] if hasattr(clf, "classes_") else []
    result: dict[str, Any] = {"caveat": CAVEAT}
    if hasattr(clf, "coef_"):
        coef = np.atleast_2d(clf.coef_)
        if coef.shape[0] == 1 and len(classes) == 2:  # binary: one vector, positive = classes_[1]
            order = np.argsort(coef[0])
            result["method"] = "linear coefficients"
            result["toward_" + classes[1]] = [(str(names[i]), float(coef[0, i])) for i in order[::-1][:n]]
            result["toward_" + classes[0]] = [(str(names[i]), float(coef[0, i])) for i in order[:n]]
        else:
            result["method"] = "linear coefficients (one-vs-rest)"
            for ci, c in enumerate(classes):
                order = np.argsort(coef[ci])[::-1][:n]
                result[c] = [(str(names[i]), float(coef[ci, i])) for i in order]
        return result
    if hasattr(clf, "feature_log_prob_"):
        lp = np.asarray(clf.feature_log_prob_)
        result["method"] = "Naive Bayes log-probability relative to the mean over classes"
        for ci, c in enumerate(classes):
            rel = lp[ci] - lp.mean(axis=0)
            order = np.argsort(rel)[::-1][:n]
            result[c] = [(str(names[i]), float(rel[i])) for i in order]
        return result
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
        order = np.argsort(imp)[::-1][:n]
        result["method"] = "impurity-based importance (global, unsigned)"
        result["overall"] = [(str(names[i]), float(imp[i])) for i in order]
        return result
    return {}
