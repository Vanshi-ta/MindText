import json

import pandas as pd
import pytest

from mindtext.config import DataConfig
from mindtext.data import load_table, resolve_positive_label, save_quality_report, validate_and_clean
from mindtext.errors import DataError


def _cfg(**kw):
    return DataConfig(task="text", **kw)


def test_missing_required_column_raises():
    with pytest.raises(DataError, match="Missing required column"):
        validate_and_clean(pd.DataFrame({"text": ["a"], "other": [1]}), _cfg())


def test_missing_file_and_bad_extension(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_table(tmp_path / "nope.csv", _cfg())
    bad = tmp_path / "x.xlsx"
    bad.write_text("x")
    with pytest.raises(DataError, match="Unsupported"):
        load_table(bad, _cfg())


def test_missing_empty_placeholder_text_dropped_and_counted():
    df = pd.DataFrame({
        "text": ["fine text", None, "   ", "[removed]", "[Deleted]", "another one"],
        "label": ["a", "a", "a", "b", "b", "b"],
    })
    out, rep = validate_and_clean(df, _cfg())
    assert list(out["text"]) == ["fine text", "another one"]
    assert rep.counts["missing_text"] == 1
    assert rep.counts["empty_text"] == 1
    assert rep.counts["placeholder_text"] == 2
    assert rep.rows_read == 6 and rep.rows_kept == 2


def test_missing_label_dropped():
    df = pd.DataFrame({"text": ["x y", "z w"], "label": ["a", None]})
    out, rep = validate_and_clean(df, _cfg())
    assert len(out) == 1 and rep.counts["missing_label"] == 1


def test_invalid_label_is_an_error_by_default_and_droppable():
    df = pd.DataFrame({"text": ["one", "two", "three"], "label": ["a", "b", "zzz"]})
    out, rep = validate_and_clean(df, _cfg(allowed_labels=["a", "b"]))
    assert rep.has_errors and any(i["code"] == "invalid_label" for i in rep.errors)
    assert "zzz" not in set(out["label"])
    _, rep2 = validate_and_clean(df, _cfg(allowed_labels=["a", "b"], invalid_label_policy="drop"))
    assert not any(i["code"] == "invalid_label" for i in rep2.errors)


def test_label_whitespace_is_normalised():
    df = pd.DataFrame({"text": ["one", "two"], "label": ["male ", "male"]})
    out, rep = validate_and_clean(df, _cfg())
    assert set(out["label"]) == {"male"} and rep.counts["label_whitespace_fixed"] == 1


@pytest.mark.parametrize("policy,expected_rows", [("report", 4), ("drop_exact", 3), ("drop_text_conflicts", 1)])
def test_dedup_policies(policy, expected_rows):
    df = pd.DataFrame({
        "text": ["same text", "Same  text", "conflict me", "Conflict me"],
        "label": ["a", "a", "a", "b"],
    })
    out, rep = validate_and_clean(df, _cfg(dedup_policy=policy))
    assert len(out) == expected_rows
    assert rep.counts["texts_with_conflicting_labels"] == 1


def test_class_distribution_and_imbalance_warning():
    df = pd.DataFrame({"text": [f"t{i} word" for i in range(40)], "label": ["a"] * 30 + ["b"] * 10})
    _, rep = validate_and_clean(df, _cfg())
    assert rep.class_distribution == {"a": 30, "b": 10}
    assert rep.imbalance_ratio == 3.0
    assert any(i["code"] == "imbalance" for i in rep.warnings)


def test_single_class_is_error():
    _, rep = validate_and_clean(pd.DataFrame({"text": ["a b", "c d"], "label": ["a", "a"]}), _cfg())
    assert any(i["code"] == "single_class" for i in rep.errors)


def test_record_ids_reproducible_and_configurable():
    df = pd.DataFrame({"id": ["x1", "x2"], "text": ["aa bb", "cc dd"], "label": ["a", "b"]})
    out, _ = validate_and_clean(df, _cfg(id_column="id"))
    assert list(out["record_id"]) == ["x1", "x2"]
    out2, _ = validate_and_clean(df.drop(columns="id"), _cfg())
    assert list(out2["record_id"]) == ["row-0", "row-1"]


def test_tabular_coercion_and_exclusions():
    n = 20
    f1 = [float(i) for i in range(n)]
    f1[1], f1[3] = "oops", float("inf")
    df = pd.DataFrame({
        "participant_id": [str(i) for i in range(n)],
        "label": ["0", "1"] * (n // 2),
        "f1": f1,
        "f2": [0.1 * i for i in range(n)],
        "Gender": ["male", "female"] * (n // 2),
        "PHQ_Score": list(range(n)),
    })
    cfg = DataConfig(task="tabular", id_column="participant_id", exclude_columns=["PHQ_Score"], positive_label="1")
    out, rep = validate_and_clean(df, cfg)
    assert rep.feature_columns == ["f1", "f2"]
    assert "Gender" in rep.excluded_columns and "PHQ_Score" not in out.columns
    assert rep.counts["malformed_numeric_cells"] == 1 and rep.counts["infinite_cells"] == 1
    assert out["f1"].isna().sum() == 2


def test_tabular_duplicate_ids_is_error():
    df = pd.DataFrame({"pid": ["1", "1", "2"], "label": ["0", "1", "0"], "f": [1.0, 2.0, 3.0]})
    _, rep = validate_and_clean(df, DataConfig(task="tabular", id_column="pid"))
    assert any(i["code"] == "duplicate_ids" for i in rep.errors)


def test_positive_label_resolution():
    assert resolve_positive_label(["0", "1"], "1") == "1"
    assert resolve_positive_label(["a", "b", "c"], None) is None
    with pytest.raises(DataError, match="positive_label"):
        resolve_positive_label(["0", "1"], None)
    with pytest.raises(DataError):
        resolve_positive_label(["0", "1"], "2")


def test_quality_report_serialises(tmp_path, text_csv):
    cfg = _cfg(id_column="id")
    out, rep = validate_and_clean(load_table(text_csv, cfg), cfg, source_path=text_csv)
    p = save_quality_report(rep, tmp_path / "q.json")
    d = json.loads(p.read_text())
    assert d["rows_kept"] == len(out) and d["file_sha256"] and d["n_errors"] == 0
