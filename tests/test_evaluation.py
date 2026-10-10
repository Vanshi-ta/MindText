import numpy as np
import pytest

from mindtext.errors import ModelError
from mindtext.evaluation import bootstrap_ci, compute_metrics, metric_value, select_threshold, trivial_baselines


def test_binary_metrics_hand_computed():
    y_true = ["1", "1", "1", "0", "0", "0", "0", "0"]
    y_pred = ["1", "1", "0", "1", "0", "0", "0", "0"]
    m = compute_metrics(y_true, y_pred, ["0", "1"], positive_label="1")
    b = m["binary"]
    assert (b["tp"], b["fn"], b["fp"], b["tn"]) == (2, 1, 1, 4)
    assert b["precision"] == pytest.approx(2 / 3) and b["recall"] == pytest.approx(2 / 3)
    assert b["f1"] == pytest.approx(2 / 3) and b["specificity"] == pytest.approx(0.8)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["confusion_matrix"]["matrix"] == [[4, 1], [1, 2]]
    assert m["per_class"]["0"]["support"] == 5


def test_positive_label_semantics_matter():
    yt, yp = ["a", "a", "a", "b"], ["a", "a", "b", "b"]
    f_b = compute_metrics(yt, yp, ["a", "b"], positive_label="b")["binary"]["f1"]
    f_a = compute_metrics(yt, yp, ["a", "b"], positive_label="a")["binary"]["f1"]
    assert f_b == pytest.approx(2 / 3) and f_a == pytest.approx(0.8)
    with pytest.raises(ModelError, match="positive_label"):
        compute_metrics(yt, yp, ["a", "b"])


def test_multiclass_macro_weighted_and_absent_class():
    yt = ["a", "a", "b", "b", "c", "c"]
    yp = ["a", "b", "b", "b", "a", "a"]
    m = compute_metrics(yt, yp, ["a", "b", "c"])
    assert m["per_class"]["c"]["recall"] == 0.0 and m["per_class"]["c"]["precision"] == 0.0  # never predicted -> 0, not error
    assert m["macro_f1"] == pytest.approx(np.mean([m["per_class"][k]["f1"] for k in "abc"]))
    assert m["weighted_f1"] == pytest.approx(np.mean([m["per_class"][k]["f1"] for k in "abc"]))  # balanced support
    assert "binary" not in m


def test_accuracy_can_hide_a_useless_model():
    yt = ["0"] * 95 + ["1"] * 5
    m = compute_metrics(yt, ["0"] * 100, ["0", "1"], positive_label="1")
    assert m["accuracy"] == 0.95 and m["binary"]["recall"] == 0.0 and m["macro_f1"] < 0.5


def test_auc_and_undefined_cases():
    m = compute_metrics(["0", "0", "1", "1"], ["0", "1", "1", "1"], ["0", "1"], y_score=[0.1, 0.6, 0.7, 0.9], positive_label="1")
    assert m["binary"]["roc_auc"] == 1.0 and m["binary"]["pr_auc"] == 1.0
    one = compute_metrics(["1", "1"], ["1", "0"], ["0", "1"], y_score=[0.2, 0.9], positive_label="1")
    assert one["binary"]["roc_auc"] is None


def test_unknown_labels_and_length_mismatch():
    with pytest.raises(ModelError, match="not in the label set"):
        compute_metrics(["a"], ["z"], ["a", "b"], positive_label="a")
    with pytest.raises(ModelError, match="differ in length"):
        compute_metrics(["a", "b"], ["a"], ["a", "b"], positive_label="a")
    with pytest.raises(ModelError, match="empty"):
        compute_metrics([], [], ["a", "b"], positive_label="a")


def test_metric_value_lookup():
    m = compute_metrics(["0", "1"], ["0", "1"], ["0", "1"], positive_label="1")
    assert metric_value(m, "macro_f1") == 1.0 and metric_value(m, "f1_positive") == 1.0
    with pytest.raises(ModelError):
        metric_value(compute_metrics(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]), "f1_positive")


def test_bootstrap_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 200).astype(str)
    yp = np.where(rng.random(200) < 0.8, yt, 1 - yt.astype(int)).astype(str)
    a = bootstrap_ci(yt, yp, ["0", "1"], "1", n_boot=300, seed=1)
    b = bootstrap_ci(yt, yp, ["0", "1"], "1", n_boot=300, seed=1)
    assert a == b
    point = compute_metrics(yt, yp, ["0", "1"], positive_label="1")
    assert a["accuracy"][0] <= point["accuracy"] <= a["accuracy"][1]
    assert a["f1_positive"][0] <= point["binary"]["f1"] <= a["f1_positive"][1]
    assert bootstrap_ci(yt, yp, ["0", "1"], "1", n_boot=0) == {}


def test_small_samples_give_wide_intervals():
    yt = np.array(["1"] * 17 + ["0"] * 39)
    yp = np.array(["1"] * 10 + ["0"] * 7 + ["1"] * 7 + ["0"] * 32)
    lo, hi = bootstrap_ci(yt, yp, ["0", "1"], "1", 2000, 0)["f1_positive"]
    assert hi - lo > 0.25


def test_select_threshold_picks_separating_value_and_avoids_degenerate_ties():
    y = ["0"] * 6 + ["1"] * 4
    s = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9]
    thr, val = select_threshold(y, s, "1", "f1_positive", default=0.5)
    assert val == 1.0 and 0.4 < thr <= 0.6
    thr_c, _ = select_threshold(y, [0.5] * 10, "1", default=0.5)
    assert thr_c == 0.5   # constant scores: no informative threshold
    with pytest.raises(ModelError, match="both classes"):
        select_threshold(["0", "0"], [0.1, 0.2], "1")


def test_trivial_baselines_match_hand_values():
    yt = ["1"] * 17 + ["0"] * 39
    b = trivial_baselines(yt, ["0", "1"], "1")
    assert b["all_positive"]["f1"] == pytest.approx(2 * (17 / 56) / (1 + 17 / 56))
    assert b["majority"]["accuracy"] == pytest.approx(39 / 56) and b["majority_class"] == "0"
