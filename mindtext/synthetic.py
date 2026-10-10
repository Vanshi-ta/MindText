"""Small SYNTHETIC datasets for demos, tests and CI.

They are generated from template words and Gaussian noise. They contain no real
posts or participants, and scores obtained on them say nothing about
real-world performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["anxiety", "depression", "lonely", "mentalhealth", "suicidewatch"]

_SIGNATURE = {
    "anxiety": ["worry", "nervous", "exam", "panic", "deadline", "overthinking"],
    "depression": ["tired", "empty", "motivation", "sleep", "numb", "heavy"],
    "lonely": ["alone", "friends", "weekend", "isolated", "quiet", "invite"],
    "mentalhealth": ["therapy", "advice", "resources", "routine", "habits", "support"],
    "suicidewatch": ["crisis", "overwhelmed", "helpline", "urgent", "exhausted", "reaching"],
}
_FILLER = ["today", "really", "think", "lately", "again", "maybe", "just", "feel", "week", "work", "people", "not", "much", "day"]
_TEMPLATES = [
    "i {f1} {s1} and {f2} {s2} {f3}",
    "{f1} {s1} {f2} about {s2} this {f3}",
    "it is {f1} {s1} when {f2} {s2} and {f3}",
    "{s1} {f1} {s2} {f2} {f3} {f1}",
]


def make_text_dataset(n_per_class: int = 40, labels: list[str] | None = None, noise: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Columns: id, text, label, group, time. ``group`` ~ pseudo-author id,
    ``time`` ~ monotonically increasing ISO date (for grouped/chronological tests)."""
    rng = np.random.default_rng(seed)
    labels = labels or LABELS
    rows = []
    for li, lab in enumerate(labels):
        sig = _SIGNATURE.get(lab, [f"{lab}{i}" for i in range(6)])
        for _ in range(n_per_class):
            pool = sig if rng.random() > noise else _SIGNATURE[labels[(li + 1) % len(labels)]] if labels[(li + 1) % len(labels)] in _SIGNATURE else sig
            tpl = _TEMPLATES[rng.integers(len(_TEMPLATES))]
            text = tpl.format(
                f1=rng.choice(_FILLER), f2=rng.choice(_FILLER), f3=rng.choice(_FILLER),
                s1=rng.choice(pool), s2=rng.choice(sig),
            )
            rows.append({"text": text, "label": lab})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    df.insert(0, "id", [f"syn-{i:05d}" for i in range(n)])
    df["group"] = [f"author-{i % max(n // 3, 1):04d}" for i in range(n)]
    df["time"] = pd.date_range("2020-01-01", periods=n, freq="6h").strftime("%Y-%m-%d %H:%M:%S")
    return df


def make_tabular_dataset(n_train: int = 90, n_dev: int = 30, n_test: int = 30, n_features: int = 12,
                         positive_rate: float = 0.25, effect: float = 0.9, seed: int = 0) -> pd.DataFrame:
    """Binary participant-level feature table with an official-style split column.
    Columns: participant_id, split, label ('0'/'1'), f00_mean ... Only the first
    three features carry signal (shift = ``effect`` standard deviations)."""
    rng = np.random.default_rng(seed)
    parts = []
    pid = 1000
    for split, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
        y = (rng.random(n) < positive_rate).astype(int)
        if y.sum() < 3:
            y[:3] = 1
        X = rng.normal(size=(n, n_features))
        X[:, :3] += effect * y[:, None]
        df = pd.DataFrame(X, columns=[f"f{i:02d}_mean" for i in range(n_features)])
        df.insert(0, "label", y.astype(str))
        df.insert(0, "split", split)
        df.insert(0, "participant_id", [str(pid + i) for i in range(n)])
        pid += n
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
