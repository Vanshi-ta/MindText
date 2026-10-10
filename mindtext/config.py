"""Centralised, validated configuration (YAML -> dataclasses)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

TASKS = ("text", "tabular")
SPLIT_STRATEGIES = ("stratified", "grouped", "chronological", "predefined")
INVALID_LABEL_POLICIES = ("error", "drop")
DEDUP_POLICIES = ("report", "drop_exact", "drop_text_conflicts")
URL_POLICIES = ("remove", "token", "keep")
TUNE_METRICS = ("f1_macro", "f1_weighted", "accuracy", "balanced_accuracy", "f1_positive")
SELECTION_METRICS = ("macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "f1_positive")
THRESHOLD_METRICS = ("f1_positive", "f1_macro", "balanced_accuracy")
MODEL_TYPES = ("majority", "nb", "logreg", "linearsvc", "svm_rbf", "rf", "xgboost", "lightgbm")


@dataclass
class DataConfig:
    path: str = "data/sample.csv"
    description: str = ""
    task: str = "text"
    text_column: str = "text"
    label_column: str = "label"
    id_column: str | None = None
    group_column: str | None = None
    time_column: str | None = None
    split_column: str | None = None
    feature_columns: list[str] | None = None
    exclude_columns: list[str] = field(default_factory=list)
    allowed_labels: list[str] | None = None
    positive_label: str | None = None
    invalid_label_policy: str = "error"
    placeholder_texts: list[str] = field(default_factory=lambda: ["[removed]", "[deleted]"])
    min_text_chars: int = 1
    dedup_policy: str = "drop_exact"


@dataclass
class SplitConfig:
    strategy: str = "stratified"
    val_size: float = 0.1
    test_size: float = 0.2
    split_names: dict[str, list[str]] = field(
        default_factory=lambda: {
            "train": ["train"],
            "val": ["dev", "val", "validation"],
            "test": ["test"],
        }
    )


@dataclass
class TextPreprocessConfig:
    lowercase: bool = True
    url_policy: str = "token"
    strip_html: bool = True
    normalize_whitespace: bool = True
    remove_stopwords: bool = False
    keep_negations: bool = True
    lemmatize: bool = False
    legacy_clean: bool = False


@dataclass
class TfidfConfig:
    max_features: int | None = 20000
    ngram_range: list[int] = field(default_factory=lambda: [1, 2])
    min_df: float = 2
    max_df: float = 1.0
    sublinear_tf: bool = True
    token_pattern: str | None = None


@dataclass
class TabularConfig:
    impute_strategy: str = "median"
    scale: bool = True
    variance_threshold: float | None = 0.0
    select_k: int | None = None


@dataclass
class ModelSpec:
    name: str = ""
    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    tune: dict[str, list[Any]] | None = None
    provenance: str = "new"  # "original" = present in the supplied notebooks; "new" = added here


@dataclass
class ThresholdConfig:
    enabled: bool = False
    metric: str = "f1_positive"
    max_candidates: int = 101


@dataclass
class TrainConfig:
    seed: int = 42
    cv_folds: int = 5
    tune_metric: str = "f1_macro"
    selection_metric: str = "macro_f1"
    bootstrap_samples: int = 200
    n_jobs: int = 1
    save_all_models: bool = False


@dataclass
class ErrorAnalysisConfig:
    n_examples: int = 10
    include_text: bool = True
    excerpt_chars: int = 160
    top_features: int = 15


@dataclass
class OutputConfig:
    reports_dir: str = "reports"
    artifacts_dir: str = "artifacts"
    run_name: str | None = None


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "split": SplitConfig,
    "text": TextPreprocessConfig,
    "tfidf": TfidfConfig,
    "tabular": TabularConfig,
    "threshold": ThresholdConfig,
    "train": TrainConfig,
    "error_analysis": ErrorAnalysisConfig,
    "output": OutputConfig,
}


def _build(cls: type, data: Any, where: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config section '{where}' must be a mapping, got {type(data).__name__}.")
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {unknown} in config section '{where}'. Allowed: {sorted(allowed)}."
        )
    return cls(**data)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    text: TextPreprocessConfig = field(default_factory=TextPreprocessConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    models: list[ModelSpec] = field(default_factory=list)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    error_analysis: ErrorAnalysisConfig = field(default_factory=ErrorAnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        if not isinstance(raw, dict):
            raise ConfigError("Top-level config must be a mapping.")
        allowed = set(_SECTIONS) | {"models"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(f"Unknown top-level config key(s): {unknown}. Allowed: {sorted(allowed)}.")
        kwargs: dict[str, Any] = {name: _build(c, raw.get(name), name) for name, c in _SECTIONS.items()}
        models_raw = raw.get("models") or []
        if not isinstance(models_raw, list):
            raise ConfigError("'models' must be a list of model specifications.")
        kwargs["models"] = [_build(ModelSpec, m, f"models[{i}]") for i, m in enumerate(models_raw)]
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse YAML config {p}: {exc}") from exc
        return cls.from_dict(raw or {})

    # ---- helpers ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        d, s, t = self.data, self.split, self.train
        if d.task not in TASKS:
            raise ConfigError(f"data.task must be one of {TASKS}, got '{d.task}'.")
        if d.invalid_label_policy not in INVALID_LABEL_POLICIES:
            raise ConfigError(f"data.invalid_label_policy must be one of {INVALID_LABEL_POLICIES}.")
        if d.dedup_policy not in DEDUP_POLICIES:
            raise ConfigError(f"data.dedup_policy must be one of {DEDUP_POLICIES}.")
        if s.strategy not in SPLIT_STRATEGIES:
            raise ConfigError(f"split.strategy must be one of {SPLIT_STRATEGIES}, got '{s.strategy}'.")
        if not (0.0 <= s.val_size < 1.0) or not (0.0 < s.test_size < 1.0):
            raise ConfigError("split.val_size must be in [0,1) and split.test_size in (0,1).")
        if s.strategy != "predefined" and s.val_size + s.test_size >= 1.0:
            raise ConfigError("split.val_size + split.test_size must be < 1.")
        if s.strategy == "grouped" and not d.group_column:
            raise ConfigError("split.strategy='grouped' requires data.group_column.")
        if s.strategy == "chronological" and not d.time_column:
            raise ConfigError("split.strategy='chronological' requires data.time_column.")
        if s.strategy == "predefined" and not d.split_column:
            raise ConfigError("split.strategy='predefined' requires data.split_column.")
        if self.text.url_policy not in URL_POLICIES:
            raise ConfigError(f"text.url_policy must be one of {URL_POLICIES}.")
        if len(self.tfidf.ngram_range) != 2 or self.tfidf.ngram_range[0] > self.tfidf.ngram_range[1]:
            raise ConfigError("tfidf.ngram_range must be [min_n, max_n] with min_n <= max_n.")
        if not self.models:
            raise ConfigError("Config must list at least one entry under 'models'.")
        names = [m.name for m in self.models]
        if any(not n for n in names):
            raise ConfigError("Every model needs a non-empty 'name'.")
        if len(set(names)) != len(names):
            raise ConfigError(f"Model names must be unique, got {names}.")
        for m in self.models:
            if m.type not in MODEL_TYPES:
                raise ConfigError(f"Model '{m.name}': unknown type '{m.type}'. Valid: {MODEL_TYPES}.")
            if m.provenance not in ("original", "new"):
                raise ConfigError(f"Model '{m.name}': provenance must be 'original' or 'new'.")
        if t.tune_metric not in TUNE_METRICS:
            raise ConfigError(f"train.tune_metric must be one of {TUNE_METRICS}.")
        if t.selection_metric not in SELECTION_METRICS:
            raise ConfigError(f"train.selection_metric must be one of {SELECTION_METRICS}.")
        if self.threshold.metric not in THRESHOLD_METRICS:
            raise ConfigError(f"threshold.metric must be one of {THRESHOLD_METRICS}.")
        if t.cv_folds < 2:
            raise ConfigError("train.cv_folds must be >= 2.")
        needs_val = len(self.models) > 1 or self.threshold.enabled
        if s.strategy != "predefined" and s.val_size == 0 and needs_val:
            raise ConfigError(
                "A validation split (split.val_size > 0) is required to select among several models "
                "or to choose a decision threshold; using the test set for either would leak it."
            )
