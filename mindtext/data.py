"""Dataset loading, validation, and data-quality reporting.

Nothing here learns from the data: all operations are row-wise checks or fixed
filters, so they are safe to run before splitting. Learned transformations
(TF-IDF, scaling, imputation) live inside scikit-learn Pipelines and are fitted
on the training split only (see ``models.py`` / ``training.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .errors import DataError

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".csv.gz", ".tsv.gz")
TINY_CLASS = 10


# --------------------------------------------------------------------------
# Quality report
# --------------------------------------------------------------------------
@dataclass
class QualityReport:
    path: str | None
    task: str
    rows_read: int = 0
    rows_kept: int = 0
    columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)
    class_proportions: dict[str, float] = field(default_factory=dict)
    imbalance_ratio: float | None = None
    constant_features: list[str] = field(default_factory=list)
    excluded_columns: dict[str, str] = field(default_factory=dict)
    file_sha256: str | None = None
    issues: list[dict[str, str]] = field(default_factory=list)

    def add(self, level: str, code: str, message: str) -> None:
        self.issues.append({"level": level, "code": code, "message": message})

    @property
    def errors(self) -> list[dict[str, str]]:
        return [i for i in self.issues if i["level"] == "error"]

    @property
    def warnings(self) -> list[dict[str, str]]:
        return [i for i in self.issues if i["level"] == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["n_errors"] = len(self.errors)
        d["n_warnings"] = len(self.warnings)
        return d

    def format_text(self) -> str:
        lines = [
            f"Data check: {self.path or '<in-memory>'}  (task={self.task})",
            f"  rows read: {self.rows_read}   rows kept: {self.rows_kept}",
        ]
        if self.file_sha256:
            lines.append(f"  file sha256: {self.file_sha256[:16]}...")
        if self.feature_columns:
            lines.append(f"  feature columns: {len(self.feature_columns)}")
        nonzero = {k: v for k, v in self.counts.items() if v}
        if nonzero:
            lines.append("  records affected: " + ", ".join(f"{k}={v}" for k, v in nonzero.items()))
        if self.class_distribution:
            lines.append("  class distribution:")
            for k, v in self.class_distribution.items():
                lines.append(f"    {k:<20} {v:>9}  ({self.class_proportions[k] * 100:5.1f}%)")
            if self.imbalance_ratio is not None:
                lines.append(f"  imbalance ratio (max/min): {self.imbalance_ratio:.2f}")
        for level in ("error", "warning"):
            for i in (x for x in self.issues if x["level"] == level):
                lines.append(f"  [{level.upper()}] {i['code']}: {i['message']}")
        if not self.issues:
            lines.append("  no issues found")
        return "\n".join(lines)


def save_quality_report(report: QualityReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return p


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _reserved_columns(cfg: DataConfig) -> list[str]:
    cols = [cfg.label_column]
    for c in (cfg.id_column, cfg.group_column, cfg.time_column, cfg.split_column):
        if c:
            cols.append(c)
    if cfg.task == "text":
        cols.append(cfg.text_column)
    return cols


def load_table(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    """Read a CSV/TSV file. Non-feature columns are read as strings so that
    labels such as ``0``/``1`` are not turned into floats by missing values."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"Dataset file not found: {p}")
    if not p.is_file():
        raise DataError(f"Dataset path is not a file: {p}")
    name = p.name.lower()
    if not name.endswith(SUPPORTED_SUFFIXES):
        raise DataError(f"Unsupported file type for {p.name}. Supported: {', '.join(SUPPORTED_SUFFIXES)}")
    sep = "\t" if ".tsv" in name else ","
    try:
        header = pd.read_csv(p, sep=sep, nrows=0)
        str_cols = [c for c in _reserved_columns(cfg) if c in header.columns]
        df = pd.read_csv(p, sep=sep, dtype={c: str for c in str_cols}, keep_default_na=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataError(f"Could not parse {p}: {exc}") from exc
    return df


# --------------------------------------------------------------------------
# Validation / cleaning
# --------------------------------------------------------------------------
def normalize_text_key(text: str) -> str:
    """Key used to detect duplicate texts (case- and whitespace-insensitive)."""
    return " ".join(text.lower().split())


def _clean_label(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    s = str(v).strip()
    return s if s else None


def validate_and_clean(
    df: pd.DataFrame, cfg: DataConfig, source_path: str | Path | None = None
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate schema and records, apply the documented drop/dedup policy.

    Returns the cleaned frame (with ``record_id`` and normalised ``label``)
    and a :class:`QualityReport`. Hard schema problems raise ``DataError``;
    record-level problems are counted and reported as issues.
    """
    rep = QualityReport(path=str(source_path) if source_path else None, task=cfg.task)
    if source_path and Path(source_path).is_file():
        rep.file_sha256 = file_sha256(source_path)
    rep.rows_read = len(df)
    rep.columns = [str(c) for c in df.columns]
    if len(df) == 0:
        raise DataError("Dataset has no rows.")

    required = [cfg.label_column] + ([cfg.text_column] if cfg.task == "text" else [])
    optional = [c for c in (cfg.id_column, cfg.group_column, cfg.time_column, cfg.split_column) if c]
    missing = [c for c in required + optional if c not in df.columns]
    if missing:
        raise DataError(f"Missing required column(s) {missing}. Columns present: {rep.columns}")

    counts = rep.counts
    out = pd.DataFrame(index=range(len(df)))

    # ---- record id (reproducible: configured id column, else file row number)
    row_ids = pd.Series([f"row-{i}" for i in range(len(df))])
    if cfg.id_column:
        raw_ids = df[cfg.id_column].reset_index(drop=True)
        ids = raw_ids.astype("string").str.strip()
        n_missing_id = int(ids.isna().sum() + (ids == "").sum())
        if n_missing_id:
            rep.add("warning", "missing_id", f"{n_missing_id} record(s) lack an id; row numbers used instead.")
        ids = ids.where(ids.notna() & (ids != ""), row_ids).astype(str)
    else:
        ids = row_ids
    out["record_id"] = ids.to_numpy()
    dup_ids = int(out["record_id"].duplicated().sum())
    counts["duplicate_record_ids"] = dup_ids
    if dup_ids:
        level = "error" if cfg.task == "tabular" else "warning"
        rep.add(level, "duplicate_ids", f"{dup_ids} duplicated record id(s) in column '{cfg.id_column}'.")

    # ---- labels
    raw_labels = df[cfg.label_column].reset_index(drop=True)
    cleaned = raw_labels.map(_clean_label)
    stripped_fixed = int(
        sum(1 for a, b in zip(raw_labels, cleaned) if isinstance(a, str) and b is not None and a != b)
    )
    counts["label_whitespace_fixed"] = stripped_fixed
    out["label"] = cleaned.to_numpy(dtype=object)
    keep = out["label"].notna().to_numpy().copy()  # writable copy (pandas may return read-only views)
    counts["missing_label"] = int((~keep).sum())
    if counts["missing_label"]:
        rep.add("warning", "missing_label", f"{counts['missing_label']} record(s) have no label and were dropped.")

    if cfg.allowed_labels is not None:
        allowed = {str(a) for a in cfg.allowed_labels}
        bad = keep & ~out["label"].isin(allowed).to_numpy()
        counts["invalid_label"] = int(bad.sum())
        if bad.any():
            examples = sorted(set(out.loc[bad, "label"]))[:5]
            level = "error" if cfg.invalid_label_policy == "error" else "warning"
            rep.add(
                level,
                "invalid_label",
                f"{int(bad.sum())} record(s) have labels outside {sorted(allowed)} (e.g. {examples}); dropped.",
            )
            keep &= ~bad
    else:
        counts["invalid_label"] = 0

    # ---- task-specific payload
    if cfg.task == "text":
        keep = _clean_text_records(df, cfg, out, keep, rep)
    else:
        keep = _clean_tabular_records(df, cfg, out, keep, rep)

    for col, name in ((cfg.group_column, "group"), (cfg.time_column, "time"), (cfg.split_column, "split")):
        if col:
            out[name] = df[col].reset_index(drop=True).to_numpy()

    result = out.loc[keep].reset_index(drop=True)
    if cfg.task == "text":
        result = _apply_dedup(result, cfg, rep)

    rep.rows_kept = len(result)
    _summarise_classes(result, cfg, rep)
    if rep.rows_kept == 0:
        raise DataError("No usable rows remain after validation.\n" + rep.format_text())
    return result, rep


def _clean_text_records(df, cfg, out, keep, rep) -> np.ndarray:
    texts = df[cfg.text_column].reset_index(drop=True)
    is_missing = texts.isna().to_numpy()
    as_str = texts.astype("string").fillna("").str.strip()
    placeholders = {p.strip().lower() for p in cfg.placeholder_texts}
    is_placeholder = as_str.str.lower().isin(placeholders).to_numpy() & ~is_missing
    too_short = (as_str.str.len() < max(cfg.min_text_chars, 1)).to_numpy() & ~is_missing
    rep.counts["missing_text"] = int((is_missing & keep).sum())
    rep.counts["empty_text"] = int((too_short & ~is_placeholder & keep).sum())
    rep.counts["placeholder_text"] = int((is_placeholder & keep).sum())
    for key, code, msg in (
        ("missing_text", "missing_text", "record(s) have missing text and were dropped"),
        ("empty_text", "empty_text", "record(s) have empty/too-short text and were dropped"),
        ("placeholder_text", "placeholder_text", f"record(s) contain placeholder text {sorted(placeholders)} and were dropped"),
    ):
        if rep.counts[key]:
            rep.add("warning", code, f"{rep.counts[key]} {msg}.")
    out["text"] = as_str.to_numpy(dtype=object)
    return keep & ~is_missing & ~too_short & ~is_placeholder


def _clean_tabular_records(df, cfg, out, keep, rep) -> np.ndarray:
    reserved = set(_reserved_columns(cfg)) | set(cfg.exclude_columns)
    if cfg.feature_columns:
        absent = [c for c in cfg.feature_columns if c not in df.columns]
        if absent:
            raise DataError(f"Configured feature column(s) not found: {absent}")
        candidates = list(cfg.feature_columns)
        explicit = True
    else:
        candidates = [c for c in df.columns if c not in reserved]
        explicit = False
    if not candidates:
        raise DataError("No feature columns available after excluding label/id/metadata columns.")

    feats: dict[str, np.ndarray] = {}
    bad_cells = 0
    for c in candidates:
        col = df[c].reset_index(drop=True)
        num = pd.to_numeric(col, errors="coerce")
        non_null = int(col.notna().sum())
        parsed = int(num.notna().sum())
        if not explicit and non_null and parsed / non_null < 0.9:
            rep.excluded_columns[str(c)] = "non-numeric column (<90% of values parse as numbers)"
            continue
        bad_cells += non_null - parsed
        feats[str(c)] = num.to_numpy(dtype=float)
    if not feats:
        raise DataError("No numeric feature columns found.")
    X = pd.DataFrame(feats)
    n_inf = int(np.isinf(X.to_numpy()).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    rep.counts["malformed_numeric_cells"] = bad_cells
    rep.counts["infinite_cells"] = n_inf
    if bad_cells:
        rep.add("warning", "malformed_numeric", f"{bad_cells} non-numeric cell(s) in feature columns treated as missing.")
    if n_inf:
        rep.add("warning", "infinite_values", f"{n_inf} infinite value(s) treated as missing.")
    if rep.excluded_columns:
        rep.add("warning", "excluded_columns", f"Excluded non-numeric columns: {sorted(rep.excluded_columns)}")

    all_nan = X.isna().all(axis=1).to_numpy()
    rep.counts["rows_all_features_missing"] = int((all_nan & keep).sum())
    if rep.counts["rows_all_features_missing"]:
        rep.add("warning", "no_features", f"{rep.counts['rows_all_features_missing']} record(s) have no usable feature values; dropped.")
    n_missing_cells = int(X.isna().sum().sum())
    rep.counts["missing_feature_cells"] = n_missing_cells
    if n_missing_cells:
        rep.add("warning", "missing_features", f"{n_missing_cells} missing feature cell(s); they are imputed inside the pipeline (train-fitted).")
    rep.constant_features = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if rep.constant_features:
        rep.add("warning", "constant_features", f"{len(rep.constant_features)} constant feature column(s) (removed by the variance filter when enabled).")
    rep.feature_columns = list(X.columns)
    for c in X.columns:
        out[c] = X[c].to_numpy()
    dup_rows = int(X.loc[keep & ~all_nan].duplicated().sum())
    rep.counts["duplicate_feature_rows"] = dup_rows
    if dup_rows:
        rep.add("warning", "duplicate_feature_rows", f"{dup_rows} record(s) have feature vectors identical to another record.")
    return keep & ~all_nan


def _apply_dedup(result: pd.DataFrame, cfg: DataConfig, rep: QualityReport) -> pd.DataFrame:
    key = result["text"].map(normalize_text_key)
    result = result.assign(_key=key)
    conflict_keys = set(result.groupby("_key")["label"].nunique().loc[lambda s: s > 1].index)
    exact_dups = int(result.duplicated(["_key", "label"]).sum())
    rep.counts["exact_duplicates"] = exact_dups
    rep.counts["texts_with_conflicting_labels"] = len(conflict_keys)
    if conflict_keys:
        rep.add("warning", "conflicting_labels", f"{len(conflict_keys)} distinct text(s) appear with more than one label.")
    if exact_dups:
        rep.add("warning", "duplicates", f"{exact_dups} duplicate (text,label) record(s) found; policy='{cfg.dedup_policy}'.")
    dropped_conflict = dropped_exact = 0
    if cfg.dedup_policy == "drop_text_conflicts" and conflict_keys:
        mask = result["_key"].isin(conflict_keys)
        dropped_conflict = int(mask.sum())
        result = result.loc[~mask]
    if cfg.dedup_policy in ("drop_exact", "drop_text_conflicts"):
        before = len(result)
        result = result.drop_duplicates(["_key", "label"], keep="first")
        dropped_exact = before - len(result)
    rep.counts["dropped_conflicting_records"] = dropped_conflict
    rep.counts["dropped_exact_duplicates"] = dropped_exact
    return result.drop(columns="_key").reset_index(drop=True)


def _summarise_classes(result: pd.DataFrame, cfg: DataConfig, rep: QualityReport) -> None:
    if len(result) == 0:
        return
    vc = result["label"].value_counts()
    rep.class_distribution = {str(k): int(v) for k, v in vc.items()}
    total = int(vc.sum())
    rep.class_proportions = {str(k): float(v) / total for k, v in vc.items()}
    rep.imbalance_ratio = float(vc.max() / vc.min()) if vc.min() > 0 else None
    if len(vc) < 2:
        rep.add("error", "single_class", f"Only one class present ({list(vc.index)}); classification needs at least two.")
    tiny = [str(k) for k, v in vc.items() if v < TINY_CLASS]
    if tiny:
        rep.add("warning", "tiny_class", f"Class(es) with fewer than {TINY_CLASS} records: {tiny}.")
    if rep.imbalance_ratio and rep.imbalance_ratio >= 3:
        rep.add("warning", "imbalance", f"Class imbalance ratio {rep.imbalance_ratio:.1f}:1 - accuracy alone is misleading; use macro-F1 / per-class recall.")
    if cfg.positive_label is not None and str(cfg.positive_label) not in vc.index:
        rep.add("error", "positive_label_missing", f"positive_label '{cfg.positive_label}' not found among labels {list(vc.index)}.")


def resolve_positive_label(labels: list[str], configured: str | None) -> str | None:
    """Binary tasks need an explicit positive class so that precision/recall
    are not computed for an arbitrary label. Multiclass tasks return None."""
    if len(labels) != 2:
        return None
    if configured is None:
        raise DataError(
            f"Binary task with labels {labels} requires data.positive_label to be set explicitly."
        )
    if str(configured) not in labels:
        raise DataError(f"positive_label '{configured}' is not one of {labels}.")
    return str(configured)
