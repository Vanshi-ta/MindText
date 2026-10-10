"""Train / validation / test splitting and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import DataConfig, SplitConfig
from .data import normalize_text_key
from .errors import DataError, LeakageError

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    seed: int

    def indices(self, name: str) -> np.ndarray:
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[name]

    def sizes(self) -> dict[str, int]:
        return {n: int(len(self.indices(n))) for n in SPLIT_NAMES}

    def class_counts(self, df: pd.DataFrame) -> dict[str, dict[str, int]]:
        return {
            n: {str(k): int(v) for k, v in df["label"].iloc[self.indices(n)].value_counts().sort_index().items()}
            for n in SPLIT_NAMES
        }

    def assignment(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [pd.DataFrame({"record_id": df["record_id"].iloc[self.indices(n)].to_numpy(), "split": n}) for n in SPLIT_NAMES]
        return pd.concat(parts, ignore_index=True)


def _wrap(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise DataError(
            f"Could not create the requested split ({exc}). Classes may be too small for the "
            "configured val/test fractions - collect more data or reduce split sizes."
        ) from exc


def _stratified(df, cfg: SplitConfig, seed: int):
    idx = np.arange(len(df))
    y = df["label"].to_numpy()
    rest, test = _wrap(train_test_split, idx, test_size=cfg.test_size, stratify=y, random_state=seed)
    if cfg.val_size > 0:
        rel = cfg.val_size / (1.0 - cfg.test_size)
        train, val = _wrap(train_test_split, rest, test_size=rel, stratify=y[rest], random_state=seed)
    else:
        train, val = rest, np.array([], dtype=int)
    return train, val, test


def _group_labels(df: pd.DataFrame) -> np.ndarray:
    arr = df["group"].astype("string").fillna("").to_numpy(dtype=object)
    for i in np.flatnonzero(arr == ""):  # records without a group are treated as their own group
        arr[i] = f"__nogroup_{i}"
    return arr


def _grouped_holdout(idx, y, groups, frac, seed):
    n_splits = max(2, int(round(1.0 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_pos, ho_pos = _wrap(lambda: next(sgkf.split(idx, y[idx], groups[idx])))
    return idx[tr_pos], idx[ho_pos]


def _grouped(df, cfg: SplitConfig, seed: int):
    idx = np.arange(len(df))
    y = df["label"].to_numpy()
    groups = _group_labels(df)
    rest, test = _grouped_holdout(idx, y, groups, cfg.test_size, seed)
    if cfg.val_size > 0:
        rel = cfg.val_size / (1.0 - cfg.test_size)
        train, val = _grouped_holdout(rest, y, groups, rel, seed)
    else:
        train, val = rest, np.array([], dtype=int)
    return train, val, test


def _chronological(df, cfg: SplitConfig):
    t = df["time"]
    num = pd.to_numeric(t, errors="coerce")
    key = num if num.notna().all() else pd.to_datetime(t, errors="coerce", utc=True)
    bad = int(pd.isna(key).sum())
    if bad:
        raise DataError(f"{bad} record(s) have an unparsable time value; cannot split chronologically.")
    order = pd.DataFrame({"k": key.to_numpy(), "r": df["record_id"].to_numpy()}).sort_values(["k", "r"], kind="mergesort").index.to_numpy()
    n = len(order)
    n_test = int(round(n * cfg.test_size))
    n_val = int(round(n * cfg.val_size))
    n_train = n - n_val - n_test
    if n_train <= 0 or n_test <= 0:
        raise DataError("Dataset too small for the requested chronological split.")
    return np.sort(order[:n_train]), np.sort(order[n_train : n_train + n_val]), np.sort(order[n_train + n_val :])


def _predefined(df, cfg: SplitConfig):
    labels = df["split"].astype(str).str.strip().str.lower()
    lookup = {alias.lower(): name for name, aliases in cfg.split_names.items() for alias in aliases}
    mapped = labels.map(lookup)
    if mapped.isna().any():
        bad = sorted(set(labels[mapped.isna()]))[:5]
        raise DataError(f"Unrecognised split value(s) {bad}; configure split.split_names. Known: {sorted(lookup)}")
    out = {n: np.flatnonzero((mapped == n).to_numpy()) for n in SPLIT_NAMES}
    if len(out["train"]) == 0 or len(out["test"]) == 0:
        raise DataError("Predefined split needs non-empty train and test partitions.")
    return out["train"], out["val"], out["test"]


def make_split(df: pd.DataFrame, cfg: SplitConfig, seed: int) -> SplitResult:
    """Create reproducible index splits (positional indices into ``df``)."""
    if cfg.strategy == "stratified":
        parts = _stratified(df, cfg, seed)
    elif cfg.strategy == "grouped":
        if "group" not in df.columns:
            raise DataError("Grouped split requires a group column.")
        parts = _grouped(df, cfg, seed)
    elif cfg.strategy == "chronological":
        if "time" not in df.columns:
            raise DataError("Chronological split requires a time column.")
        parts = _chronological(df, cfg)
    elif cfg.strategy == "predefined":
        if "split" not in df.columns:
            raise DataError("Predefined split requires a split column.")
        parts = _predefined(df, cfg)
    else:  # pragma: no cover - guarded by Config.validate
        raise DataError(f"Unknown split strategy {cfg.strategy}")
    tr, va, te = (np.sort(np.asarray(p, dtype=int)) for p in parts)
    return SplitResult(tr, va, te, cfg.strategy, seed)


# --------------------------------------------------------------------------
# Leakage checks
# --------------------------------------------------------------------------
def _row_keys(df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[feature_columns].round(10), index=False).astype(str)


def leakage_report(
    df: pd.DataFrame,
    split: SplitResult,
    data_cfg: DataConfig,
    feature_columns: list[str] | None = None,
) -> dict:
    """Verify separation of the three partitions. Raises ``LeakageError`` for
    hard violations (shared rows, or shared groups under a grouped split) and
    otherwise returns counts that are stored in the run report."""
    names = list(SPLIT_NAMES)
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    report: dict = {"index_overlap": {}, "record_id_overlap": {}, "group_overlap": None, "cross_split_duplicates": {}}

    for a, b in pairs:
        key = f"{a}|{b}"
        report["index_overlap"][key] = int(len(np.intersect1d(split.indices(a), split.indices(b))))
        ia, ib = set(df["record_id"].iloc[split.indices(a)]), set(df["record_id"].iloc[split.indices(b)])
        report["record_id_overlap"][key] = len(ia & ib)
    if any(report["index_overlap"].values()) or any(report["record_id_overlap"].values()):
        raise LeakageError(f"Rows shared between partitions: {report['index_overlap']} / ids {report['record_id_overlap']}")

    if "group" in df.columns:
        groups = {n: set(df["group"].iloc[split.indices(n)].dropna().astype(str)) for n in names}
        report["group_overlap"] = {f"{a}|{b}": len(groups[a] & groups[b]) for a, b in pairs}
        if split.strategy == "grouped" and any(report["group_overlap"].values()):
            raise LeakageError(f"Grouped split shares groups across partitions: {report['group_overlap']}")

    if data_cfg.task == "text":
        keys = df["text"].map(normalize_text_key)
    elif feature_columns:
        keys = _row_keys(df, feature_columns)
    else:
        keys = None
    if keys is not None:
        sets = {n: set(keys.iloc[split.indices(n)]) for n in names}
        report["cross_split_duplicates"] = {f"{a}|{b}": len(sets[a] & sets[b]) for a, b in pairs}
    report["clean"] = not any(report["cross_split_duplicates"].values()) and not any((report["group_overlap"] or {}).values())
    return report
