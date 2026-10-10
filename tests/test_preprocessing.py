import importlib.util

import numpy as np
import pytest
from sklearn.base import clone

from mindtext.preprocessing import STATS, TextCleaner, aggregate_frames, aggregated_names

HAS_NLTK = importlib.util.find_spec("nltk") is not None


def test_default_cleaner_keeps_negations_punctuation_emoji_digits():
    out = TextCleaner().transform(["I do NOT feel okay!! :( 😞 3 days"])[0]
    assert out == "i do not feel okay!! :( 😞 3 days"


def test_url_policies():
    t = ["see https://example.com/x?y=1 now"]
    assert TextCleaner(url_policy="token").transform(t) == ["see urltoken now"]
    assert TextCleaner(url_policy="remove").transform(t) == ["see now"]
    assert "https://example.com" in TextCleaner(url_policy="keep").transform(t)[0]


def test_html_and_whitespace():
    assert TextCleaner().transform(["a &amp; b <b>bold</b>\n\n  c"]) == ["a & b bold c"]


def test_stopword_removal_keeps_negations_by_default():
    out = TextCleaner(remove_stopwords=True).transform(["this is not good and I am never okay"])[0]
    assert "not" in out.split() and "never" in out.split() and "this" not in out.split()
    out2 = TextCleaner(remove_stopwords=True, keep_negations=False).transform(["this is not good"])[0]
    assert "not" not in out2.split()


def test_handles_none_nan_and_rejects_bare_string():
    assert TextCleaner().transform([None, float("nan"), 5]) == ["", "", "5"]
    with pytest.raises(TypeError):
        TextCleaner().transform("a single string")


def test_cleaner_is_stateless_and_clonable():
    c = TextCleaner(lowercase=False)
    a = c.fit(["x"]).transform(["Hello"])
    b = clone(c).fit(["completely different data"]).transform(["Hello"])
    assert a == b == ["Hello"]


@pytest.mark.skipif(HAS_NLTK, reason="only checks the no-nltk error path")
def test_legacy_mode_without_nltk_gives_clear_error():
    from mindtext.errors import MissingDependencyError
    with pytest.raises(MissingDependencyError, match="nltk"):
        TextCleaner(legacy_clean=True).fit(["x"])


def test_aggregate_frames_shape_order_and_values():
    frames = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    v = aggregate_frames(frames)
    assert v.shape == (2 * len(STATS),)
    assert v[0] == 2.5 and v[1] == 25.0            # mean block first (stat-major order)
    assert v[2 * 3] == 1.0                           # min of feature 0
    assert aggregated_names(["a", "b"], STATS[:2], "eg") == ["eg_a_mean", "eg_b_mean", "eg_a_std", "eg_b_std"]


def test_aggregate_frames_sentinel_and_nan_fill():
    frames = np.array([[-201.0, 1.0], [-201.0, 3.0], [5.0, np.inf]])
    kept = aggregate_frames(frames, ("mean",), sentinels=(), nan_fill=None)
    masked = aggregate_frames(frames, ("mean",), sentinels=(-201.0,))
    assert kept[0] == pytest.approx((-201 - 201 + 5) / 3)   # the sentinel biases the statistic
    assert masked[0] == 5.0 and masked[1] == 2.0
    allbad = aggregate_frames(np.full((3, 1), -201.0), ("mean", "std"), sentinels=(-201.0,))
    assert np.isnan(allbad).all()
    assert (aggregate_frames(np.full((3, 1), -201.0), ("mean",), (-201.0,), nan_fill=0.0) == 0).all()


def test_aggregate_frames_validation():
    with pytest.raises(ValueError):
        aggregate_frames(np.zeros(5))
    with pytest.raises(ValueError):
        aggregate_frames(np.zeros((2, 2)), ("variance",))
