import pandas as pd

from mindtext.error_analysis import collect_errors, excerpt, redact, top_features
from mindtext.models import build_pipeline


def test_redaction_removes_identifying_patterns():
    t = "mail me at jane.doe@example.com or call 555 123 4567, see https://x.org/a and ask u/some_user or @handle"
    r = redact(t)
    for secret in ("jane.doe", "555 123", "x.org", "some_user", "handle"):
        assert secret not in r
    assert "[EMAIL]" in r and "[URL]" in r and "[USER]" in r and "[NUM]" in r


def test_excerpt_truncates():
    assert len(excerpt("word " * 100, 40)) <= 40 and excerpt("word " * 100, 40).endswith("...")


def _preds():
    return pd.DataFrame({
        "record_id": [f"r{i}" for i in range(8)],
        "y_true": ["1", "1", "1", "0", "0", "0", "0", "1"],
        "y_pred": ["1", "0", "0", "1", "1", "0", "0", "1"],
        "score": [0.9, 0.2, 0.4, 0.8, 0.6, 0.1, 0.3, 0.7],
        "score_kind": "proba",
    })


def test_binary_false_positives_and_negatives_ranked_by_confidence():
    texts = {f"r{i}": f"text {i} see https://a.b/c" for i in range(8)}
    e = collect_errors(_preds(), "1", texts, n_examples=5)
    assert [x["record_id"] for x in e["false_positives"]] == ["r3", "r4"]      # highest score first
    assert [x["record_id"] for x in e["false_negatives"]] == ["r1", "r2"]      # lowest score first
    assert e["n_false_positives"] == 2 and e["n_false_negatives"] == 2 and e["n_errors"] == 4
    item = e["false_positives"][0]
    assert item["actual"] == "0" and item["predicted"] == "1" and item["score"] == 0.8 and "[URL]" in item["excerpt"]


def test_text_can_be_omitted():
    e = collect_errors(_preds(), "1", {"r1": "secret"}, include_text=False)
    assert all("excerpt" not in x for x in e["false_negatives"])


def test_multiclass_confusions():
    p = pd.DataFrame({"record_id": list("abcdef"), "y_true": ["x", "x", "x", "y", "y", "z"],
                      "y_pred": ["y", "y", "x", "x", "y", "z"], "score": [.9, .8, .7, .6, .5, .4], "score_kind": "proba"})
    e = collect_errors(p)
    top = e["top_confusions"][0]
    assert (top["actual"], top["predicted"], top["count"]) == ("x", "y", 2)


def test_top_terms_for_linear_model(text_cfg, text_df):
    pipe = build_pipeline(text_cfg, text_cfg.models[1]).fit(text_df["text"], text_df["label"])
    tf = top_features(pipe, 5)
    assert "caveat" in tf and "not causal" in tf["caveat"]
    assert any(w in dict(tf["anxiety"]) for w in ("worry", "nervous", "panic", "exam", "deadline", "overthinking"))


def test_top_features_unsupported_returns_empty(text_cfg, text_df):
    maj = build_pipeline(text_cfg, text_cfg.models[0]).fit(text_df["text"], text_df["label"])
    assert top_features(maj) == {}
