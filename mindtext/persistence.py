"""Model persistence with integrity metadata.

SECURITY: joblib/pickle files can execute arbitrary code when loaded. Only load
model files you created yourself or obtained from a source you trust. The
SHA-256 sidecar protects against accidental corruption and mix-ups; it does NOT
protect against an attacker who can replace both files.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .errors import IntegrityError, ModelError
from .models import ModelBundle

SECURITY_NOTE = (
    "Serialized with joblib (pickle). Load only from trusted sources; checksum verification does not "
    "make an untrusted file safe."
)


def meta_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_name(p.name + ".meta.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mindtext": __version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def save_bundle(bundle: ModelBundle, path: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p, compress=3)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": _sha256(p),
        "size_bytes": p.stat().st_size,
        "task": bundle.task,
        "labels": bundle.labels,
        "positive_label": bundle.positive_label,
        "threshold": bundle.threshold,
        "text_column": bundle.text_column,
        "n_feature_columns": len(bundle.feature_columns or []),
        "environment": environment_info(),
        "security_note": SECURITY_NOTE,
        "bundle_metadata": bundle.metadata,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path(p).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return p


def load_bundle(path: str | Path, verify: bool = True) -> ModelBundle:
    p = Path(path)
    if not p.exists():
        raise ModelError(f"Model file not found: {p}")
    mp = meta_path(p)
    meta: dict[str, Any] | None = None
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if verify and meta.get("sha256") != _sha256(p):
            raise IntegrityError(
                f"Checksum mismatch for {p.name}: the file differs from what was recorded in {mp.name}. "
                "Refusing to load."
            )
        saved = (meta.get("environment") or {}).get("scikit-learn")
        if saved and saved != sklearn.__version__:
            warnings.warn(
                f"Model was saved with scikit-learn {saved}; running {sklearn.__version__}. "
                "Predictions may differ or loading may fail.",
                stacklevel=2,
            )
    elif verify:
        warnings.warn(f"No metadata sidecar found for {p.name}; integrity cannot be verified.", stacklevel=2)
    obj = joblib.load(p)
    if not isinstance(obj, ModelBundle):
        raise ModelError(f"{p} does not contain a MindText model bundle (found {type(obj).__name__}).")
    return obj
