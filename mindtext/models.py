"""Model registry, scikit-learn Pipeline construction, and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

from .config import Config, ModelSpec
from .errors import MissingDependencyError, ModelError
from .preprocessing import TextCleaner

NEEDS_SCALING = {"logreg", "linearsvc", "svm_rbf"}


# --------------------------------------------------------------------------
# Wrapper for libraries that require integer class labels (e.g. XGBoost)
# --------------------------------------------------------------------------
class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Encode arbitrary labels to 0..K-1 for the wrapped estimator.

    If the wrapped estimator has ``scale_pos_weight='auto'`` it is replaced by
    n_negative / n_positive computed from the *training data seen by this fit*
    (so inside cross-validation it uses the training fold only).
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        y_enc = self.encoder_.transform(y)
        est = clone(self.estimator)
        if getattr(est, "scale_pos_weight", None) == "auto":
            if len(self.classes_) != 2:
                raise ModelError("scale_pos_weight='auto' is only valid for binary problems.")
            pos = max(int((y_enc == 1).sum()), 1)
            est.set_params(scale_pos_weight=float((y_enc == 0).sum()) / pos)
        self.estimator_ = est.fit(X, y_enc)
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(np.asarray(self.estimator_.predict(X)).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def _make_classifier(spec: ModelSpec, task: str, seed: int, n_jobs: int):
    t, p = spec.type, dict(spec.params)
    if t == "majority":
        return DummyClassifier(strategy="most_frequent", **p)
    if t == "nb":
        if task != "text":
            raise ModelError("Multinomial Naive Bayes needs non-negative features; use it for task 'text' only.")
        return MultinomialNB(**p)
    if t == "logreg":
        return LogisticRegression(**{"max_iter": 1000, "random_state": seed, **p})
    if t == "linearsvc":
        return LinearSVC(**{"random_state": seed, **p})
    if t == "svm_rbf":
        # probability=False on purpose: Platt scaling collapsed on the small
        # E-DAIC splits (see docs/original_results.md); the margin is used instead.
        return SVC(**{"kernel": "rbf", "probability": False, "random_state": seed, **p})
    if t == "rf":
        return RandomForestClassifier(**{"n_estimators": 100, "random_state": seed, "n_jobs": n_jobs, **p})
    if t == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise MissingDependencyError("Model type 'xgboost' needs: pip install 'mindtext[boosting]'.") from exc
        base = XGBClassifier(**{"random_state": seed, "n_jobs": n_jobs, "eval_metric": "logloss", "verbosity": 0, **p})
        return LabelEncodedClassifier(base)
    if t == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise MissingDependencyError("Model type 'lightgbm' needs: pip install 'mindtext[boosting]'.") from exc
        return LGBMClassifier(**{"random_state": seed, "n_jobs": n_jobs, "verbose": -1, **p})
    raise ModelError(f"Unknown model type '{t}'.")  # pragma: no cover


def build_pipeline(cfg: Config, spec: ModelSpec) -> Pipeline:
    """Build an unfitted Pipeline. Every learned step (vectorizer, imputer,
    scaler, selector, classifier) is inside the Pipeline, so ``fit`` on the
    training split / CV training folds is the only place statistics are learned."""
    seed, n_jobs = cfg.train.seed, cfg.train.n_jobs
    clf = _make_classifier(spec, cfg.data.task, seed, n_jobs)
    if cfg.data.task == "text":
        t = cfg.tfidf
        tfidf = TfidfVectorizer(
            lowercase=False,  # lower-casing is controlled by TextCleaner
            max_features=t.max_features,
            ngram_range=tuple(t.ngram_range),
            min_df=t.min_df,
            max_df=t.max_df,
            sublinear_tf=t.sublinear_tf,
            **({"token_pattern": t.token_pattern} if t.token_pattern else {}),
        )
        steps = [("clean", TextCleaner(**cfg.text.__dict__)), ("tfidf", tfidf), ("clf", clf)]
    else:
        tb = cfg.tabular
        steps = [("impute", SimpleImputer(strategy=tb.impute_strategy, keep_empty_features=True))]
        if tb.variance_threshold is not None:
            steps.append(("variance", VarianceThreshold(threshold=tb.variance_threshold)))
        if tb.scale and spec.type in NEEDS_SCALING:
            steps.append(("scale", StandardScaler()))
        if tb.select_k:
            steps.append(("select", SelectKBest(f_classif, k=tb.select_k)))
        steps.append(("clf", clf))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Scores / decisions
# --------------------------------------------------------------------------
def default_threshold(kind: str) -> float:
    return 0.5 if kind == "proba" else 0.0


def positive_scores(pipe, X, positive_label: str) -> tuple[np.ndarray, str]:
    """Score for the positive class. Higher = more positive.
    Uses ``predict_proba`` when the model has it, else ``decision_function``
    (an uncalibrated margin, not a probability)."""
    classes = [str(c) for c in pipe.classes_]
    pos = classes.index(str(positive_label))
    if hasattr(pipe, "predict_proba"):
        return np.asarray(pipe.predict_proba(X))[:, pos], "proba"
    if hasattr(pipe, "decision_function"):
        d = np.asarray(pipe.decision_function(X))
        if d.ndim == 1:  # sklearn: positive class is classes_[1]
            return (d if pos == 1 else -d), "decision"
        return d[:, pos], "decision"
    raise ModelError("Model exposes neither predict_proba nor decision_function.")


def confidence_scores(pipe, X) -> tuple[np.ndarray | None, str | None]:
    """Multiclass confidence: max class probability, or max decision margin."""
    if hasattr(pipe, "predict_proba"):
        return np.asarray(pipe.predict_proba(X)).max(axis=1), "proba"
    if hasattr(pipe, "decision_function"):
        d = np.asarray(pipe.decision_function(X))
        return (np.abs(d) if d.ndim == 1 else d.max(axis=1)), "decision"
    return None, None


def predict_with_scores(pipe, X, positive_label: str | None, threshold: float | None):
    """Return (predictions, score, score_kind). Binary tasks with a threshold
    use ``score >= threshold`` for the positive class; otherwise ``predict``."""
    classes = [str(c) for c in pipe.classes_]
    if positive_label is not None and len(classes) == 2:
        score, kind = positive_scores(pipe, X, positive_label)
        if threshold is None:
            pred = np.asarray(pipe.predict(X)).astype(str)
        else:
            neg = [c for c in classes if c != str(positive_label)][0]
            pred = np.where(score >= threshold, str(positive_label), neg)
        return pred, score, kind
    score, kind = confidence_scores(pipe, X)
    return np.asarray(pipe.predict(X)).astype(str), score, kind


# --------------------------------------------------------------------------
# Deployable bundle
# --------------------------------------------------------------------------
@dataclass
class ModelBundle:
    pipeline: Any
    task: str
    labels: list[str]
    positive_label: str | None = None
    threshold: float | None = None
    text_column: str | None = None
    feature_columns: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_frame(self, X) -> pd.DataFrame:
        """Predict for a list of texts (task 'text') or a DataFrame holding the
        training feature columns (task 'tabular')."""
        if self.task == "text":
            if isinstance(X, pd.DataFrame):
                raise ModelError("Text models take a sequence of strings, not a DataFrame.")
            data: Any = list(X)
        else:
            if not isinstance(X, pd.DataFrame):
                raise ModelError("Tabular models take a DataFrame containing the training feature columns.")
            missing = [c for c in (self.feature_columns or []) if c not in X.columns]
            if missing:
                raise ModelError(f"Input is missing {len(missing)} feature column(s), e.g. {missing[:5]}")
            data = X[self.feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if len(data) == 0:
            raise ModelError("No input records to predict.")
        pred, score, kind = predict_with_scores(self.pipeline, data, self.positive_label, self.threshold)
        out = pd.DataFrame({"prediction": pred, "score": score, "score_kind": kind})
        if hasattr(self.pipeline, "predict_proba"):
            proba = np.asarray(self.pipeline.predict_proba(data))
            for i, c in enumerate(self.pipeline.classes_):
                out[f"proba_{c}"] = proba[:, i]
        return out


def feature_names_after_selection(pipe: Pipeline, input_names: list[str]) -> list[str]:
    """Track feature names through variance / select steps (tabular pipelines)."""
    names = np.asarray(input_names, dtype=object)
    for step in ("variance", "select"):
        if step in pipe.named_steps:
            names = names[pipe.named_steps[step].get_support()]
    return list(names)
