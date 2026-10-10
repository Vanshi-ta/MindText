"""Stateless preprocessing.

``TextCleaner`` has no learned state, so applying it before or inside a
cross-validation loop cannot leak information. It is nevertheless used as the
first step of a Pipeline so that training and inference are identical.

Design note: the original notebooks lower-cased, stripped every non-letter
(digits, punctuation, emoji, apostrophes), and removed NLTK stop-words, which
include negations such as "not" and "no". Those choices are *not* the default
here; they are available as ``legacy_clean`` for replication only.
"""

from __future__ import annotations

import html
import re
import warnings
from typing import Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .errors import MissingDependencyError

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
LEGACY_URL_RE = re.compile(r"http\S+|www\S+|https\S+")
LEGACY_NONLETTER_RE = re.compile(r"[^a-z\s]")

NEGATIONS = frozenset(
    {"no", "not", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot", "cant", "without"}
)
_TOKEN_STRIP = ".,;:!?\"'()[]{}*_~`"


def _as_strings(X: Iterable) -> list[str]:
    if isinstance(X, str):
        raise TypeError("Expected a sequence of strings, got a single string. Wrap it in a list.")
    return ["" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in X]


class TextCleaner(BaseEstimator, TransformerMixin):
    """Configurable, stateless text normaliser."""

    def __init__(
        self,
        lowercase: bool = True,
        url_policy: str = "token",
        strip_html: bool = True,
        normalize_whitespace: bool = True,
        remove_stopwords: bool = False,
        keep_negations: bool = True,
        lemmatize: bool = False,
        legacy_clean: bool = False,
    ):
        self.lowercase = lowercase
        self.url_policy = url_policy
        self.strip_html = strip_html
        self.normalize_whitespace = normalize_whitespace
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        self.lemmatize = lemmatize
        self.legacy_clean = legacy_clean

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        if self.url_policy not in ("remove", "token", "keep"):
            raise ValueError("url_policy must be 'remove', 'token' or 'keep'.")
        if self.legacy_clean:
            _nltk_resources()  # fail early with a clear message
        elif self.lemmatize:
            _get_lemmatizer()
        return self

    def transform(self, X):
        texts = _as_strings(X)
        if self.legacy_clean:
            return [legacy_clean_text(t) for t in texts]
        stop = None
        if self.remove_stopwords:
            stop = set(ENGLISH_STOP_WORDS)
            if self.keep_negations:
                stop -= NEGATIONS
        lemmatizer = _get_lemmatizer() if self.lemmatize else None
        return [self._clean_one(t, stop, lemmatizer) for t in texts]

    def _clean_one(self, text: str, stop, lemmatizer) -> str:
        if self.strip_html:
            text = HTML_TAG_RE.sub(" ", html.unescape(text))
        if self.url_policy == "remove":
            text = URL_RE.sub(" ", text)
        elif self.url_policy == "token":
            text = URL_RE.sub(" urltoken ", text)
        if self.lowercase:
            text = text.lower()
        if stop is not None or lemmatizer is not None:
            tokens = []
            for tok in text.split():
                core = tok.strip(_TOKEN_STRIP).lower()
                if stop is not None and core in stop:
                    continue
                if lemmatizer is not None and core.isalpha():
                    tok = lemmatizer.lemmatize(core)  # surrounding punctuation is dropped in this mode
                tokens.append(tok)
            text = " ".join(tokens)
        if self.normalize_whitespace:
            text = WS_RE.sub(" ", text).strip()
        return text

    def __sklearn_tags__(self):  # keep sklearn>=1.6 happy for string input
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        return tags


def _get_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise MissingDependencyError("Lemmatization requires nltk: pip install 'mindtext[nlp]'.") from exc
    lem = WordNetLemmatizer()
    try:
        lem.lemmatize("tests")
    except LookupError as exc:
        raise MissingDependencyError(
            "NLTK 'wordnet' data not found. Run: python -c \"import nltk; nltk.download('wordnet')\""
        ) from exc
    return lem


def _nltk_resources():
    try:
        from nltk.corpus import stopwords
    except ImportError as exc:
        raise MissingDependencyError("legacy_clean needs nltk: pip install 'mindtext[nlp]'.") from exc
    try:
        words = set(stopwords.words("english"))
    except LookupError as exc:
        raise MissingDependencyError(
            "NLTK 'stopwords' data not found. Run: python -c \"import nltk; nltk.download('stopwords')\""
        ) from exc
    return words, _get_lemmatizer()


def legacy_clean_text(text: str) -> str:
    """Re-implementation of ``clean_text`` from the original Reddit notebooks
    (lower-case; drop URLs; keep only a-z and spaces; drop NLTK stop-words and
    tokens shorter than 3 characters; WordNet-lemmatise). Because only letters
    and spaces survive, whitespace splitting is equivalent to the notebook's
    ``word_tokenize``. Provided for replication, NOT recommended: it removes
    negations, digits, punctuation, and emoji."""
    stop_words, lemmatizer = _nltk_resources()
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = LEGACY_URL_RE.sub("", text)
    text = LEGACY_NONLETTER_RE.sub("", text)
    return " ".join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2)


# --------------------------------------------------------------------------
# Acoustic frame -> participant-level aggregation (stateless, per participant)
# --------------------------------------------------------------------------
STATS = ("mean", "std", "median", "min", "max", "p25", "p75")


def aggregate_frames(
    frames: np.ndarray,
    stats: Sequence[str] = STATS,
    sentinels: Sequence[float] = (),
    nan_fill: float | None = None,
) -> np.ndarray:
    """Summarise a (n_frames, n_features) matrix into a stat-major vector
    ``[stat0_f0..stat0_fk, stat1_f0..]``.

    ``sentinels``: values (e.g. openSMILE's -201) to treat as missing.
    ``nan_fill``: replacement for statistics that are still NaN (all frames
    missing). ``None`` keeps NaN so a train-fitted imputer can handle it; the
    original notebooks used 0.0.

    Each participant is processed independently and no label is used, so this
    step cannot leak information between participants or splits.
    """
    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"frames must be 2-D (n_frames, n_features); got shape {arr.shape}")
    unknown = [s for s in stats if s not in STATS]
    if unknown:
        raise ValueError(f"Unknown statistic(s) {unknown}; valid: {STATS}")
    arr = np.where(np.isfinite(arr), arr, np.nan)
    for s in sentinels:
        arr = np.where(arr == s, np.nan, arr)
    funcs = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
        "p25": lambda a, axis: np.nanpercentile(a, 25, axis=axis),
        "p75": lambda a, axis: np.nanpercentile(a, 75, axis=axis),
    }
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        for s in stats:
            parts.append(funcs[s](arr, axis=0))
    vec = np.concatenate(parts)
    if nan_fill is not None:
        vec = np.where(np.isfinite(vec), vec, nan_fill)
    return vec


def aggregated_names(feature_names: Sequence[str], stats: Sequence[str] = STATS, prefix: str = "") -> list[str]:
    pre = f"{prefix}_" if prefix else ""
    return [f"{pre}{f}_{s}" for s in stats for f in feature_names]
