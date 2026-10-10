"""Convert raw third-party files into the tidy CSVs the pipeline expects.

Both functions only reshape data. They never read labels to compute features and
never fit anything, so they are safe to run before splitting. The raw datasets
are NOT bundled with this repository (see data/README.md).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import DataError
from .preprocessing import STATS, aggregate_frames, aggregated_names

# Ordered (regex on lower-cased file stem, label). Derived from file names seen in the
# original notebook log (depjan20, lonenov20, SWjan20, Mhjan20, anxijan20, Mheaug20, ...).
# The original code only matched 3-letter prefixes 'ani','dep','lon','mhj','swj', so files
# such as swnov20, mhov20, anxinov20, Swaug20 copy, Mheaug20 fell into 'unknown' and were
# silently discarded. The rules below are explicit and unmatched files are reported.
DEFAULT_REDDIT_RULES: list[tuple[str, str]] = [
    (r"^dep", "depression"),
    (r"^lon", "lonely"),
    (r"^sw", "suicidewatch"),
    (r"^mh", "mentalhealth"),
    (r"^anx", "anxiety"),
]


def infer_label(filename: str, rules=DEFAULT_REDDIT_RULES) -> str | None:
    stem = Path(filename).stem.strip().lower()
    for pattern, label in rules:
        if re.search(pattern, stem):
            return label
    return None


def _hash_group(value, salt: str) -> str:
    return hashlib.sha256((salt + str(value)).encode("utf-8")).hexdigest()[:16]


def prepare_reddit(
    raw_dir: str | Path,
    out_csv: str | Path,
    text_column: str = "selftext",
    rules=DEFAULT_REDDIT_RULES,
    strict: bool = False,
    hash_salt: str = "",
) -> dict:
    """Walk ``raw_dir`` for CSV files, label each by file name, and write a
    combined CSV with columns text,label,source_file (+ id/time/group if the
    source has id / created_utc / author columns - names assumed, unverified).
    Author names are replaced by a salted SHA-256 prefix (pseudonymous, not anonymous)."""
    raw = Path(raw_dir)
    if not raw.is_dir():
        raise DataError(f"Raw Reddit directory not found: {raw}")
    files = sorted(p for p in raw.rglob("*.csv"))
    if not files:
        raise DataError(f"No CSV files found under {raw}")
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {"files": [], "unmatched_files": [], "suspected_copies": [], "rows_written": 0, "per_label": {}}
    first = True
    for f in files:
        label = infer_label(f.name, rules)
        rel = f.relative_to(raw).as_posix()
        if label is None:
            summary["unmatched_files"].append(rel)
            continue
        if re.search(r"\bcopy\b", f.stem.lower()):
            summary["suspected_copies"].append(rel)
        try:
            df = pd.read_csv(f, dtype=str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise DataError(f"Could not read {rel}: {exc}") from exc
        if text_column not in df.columns:
            summary["files"].append({"file": rel, "label": label, "rows": 0, "skipped": f"no '{text_column}' column"})
            continue
        tidy = pd.DataFrame({"text": df[text_column], "label": label, "source_file": rel})
        if "id" in df.columns:
            tidy["id"] = df["id"]
        if "created_utc" in df.columns:
            tidy["time"] = df["created_utc"]
        if "author" in df.columns:
            tidy["group"] = df["author"].map(lambda v: _hash_group(v, hash_salt) if pd.notna(v) else None)
        tidy.to_csv(out, mode="w" if first else "a", header=first, index=False)
        first = False
        summary["files"].append({"file": rel, "label": label, "rows": int(len(tidy))})
        summary["rows_written"] += int(len(tidy))
        summary["per_label"][label] = summary["per_label"].get(label, 0) + int(len(tidy))
    if summary["unmatched_files"] and strict:
        raise DataError(f"{len(summary['unmatched_files'])} file(s) match no label rule: {summary['unmatched_files'][:5]}")
    if first:
        raise DataError("No rows were written; check file names and --text-column.")
    return summary


FEATURE_SOURCES = {"egemaps": ("*egemaps.csv", "eg"), "mfcc": ("*mfcc.csv", "mfcc")}


def _find_frame_file(part_dir: Path, pattern: str) -> Path | None:
    hits = sorted(h for h in part_dir.rglob(pattern) if "boaw" not in h.name.lower())
    return hits[0] if hits else None


def build_acoustic_features(
    raw_dir: str | Path,
    labels_dir: str | Path,
    out_csv: str | Path,
    sources: tuple[str, ...] = ("egemaps",),
    stats: tuple[str, ...] = STATS,
    sentinels: tuple[float, ...] = (-201.0,),
    nan_fill: float | None = None,
) -> dict:
    """Aggregate frame-level openSMILE files into one row per participant.

    Expects ``labels_dir/{train,dev,test}_split.csv`` with Participant_ID and
    PHQ_Binary, and ``raw_dir/<id>_P/*egemaps.csv`` (and ``*mfcc.csv``) written
    with ';' separators. Output columns: participant_id, split, label, features.
    PHQ_Score and other metadata are deliberately NOT written: PHQ_Binary is derived
    from the PHQ score, so keeping it next to the features invites target leakage.
    Defaults differ from the first notebooks (which kept the -201 sentinel as data)
    - see docs/original_results.md.
    """
    raw, lab = Path(raw_dir), Path(labels_dir)
    for d in (raw, lab):
        if not d.is_dir():
            raise DataError(f"Directory not found: {d}")
    unknown = [s for s in sources if s not in FEATURE_SOURCES]
    if unknown:
        raise DataError(f"Unknown feature source(s) {unknown}; valid: {sorted(FEATURE_SOURCES)}")
    rows, missing = [], []
    splits_seen = {}
    for split in ("train", "dev", "test"):
        f = lab / f"{split}_split.csv"
        if not f.is_file():
            raise DataError(f"Label file not found: {f}")
        ldf = pd.read_csv(f)
        for col in ("Participant_ID", "PHQ_Binary"):
            if col not in ldf.columns:
                raise DataError(f"{f.name} lacks column '{col}'. Columns: {list(ldf.columns)}")
        splits_seen[split] = len(ldf)
        for pid, y in zip(ldf["Participant_ID"], ldf["PHQ_Binary"]):
            pid = int(pid)
            vec_parts, names = [], []
            ok = True
            for src in sources:
                pattern, prefix = FEATURE_SOURCES[src]
                fp = _find_frame_file(raw / f"{pid}_P", pattern)
                if fp is None:
                    ok = False
                    break
                fr = pd.read_csv(fp, sep=";")
                cols = [c for c in fr.columns if c not in ("name", "frameTime")]
                arr = fr[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                vec_parts.append(aggregate_frames(arr, stats, sentinels, nan_fill))
                names += aggregated_names(cols, stats, prefix)
            if not ok:
                missing.append(pid)
                continue
            row = dict(zip(names, np.concatenate(vec_parts)))
            rows.append({"participant_id": str(pid), "split": split, "label": str(int(y)), **row})
    if not rows:
        raise DataError("No participants could be processed; check raw_dir layout '<id>_P/*egemaps.csv'.")
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return {"rows": len(out), "n_features": out.shape[1] - 3, "per_split": splits_seen, "missing_participants": missing,
            "sentinels": list(sentinels), "stats": list(stats), "sources": list(sources)}
