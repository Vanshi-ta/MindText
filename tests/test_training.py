import json

import numpy as np
import pandas as pd
import pytest

from mindtext.errors import DataError
from mindtext.evaluation import compute_metrics, select_threshold
from mindtext.persistence import load_bundle
from mindtext.training import run_experiment


def _quiet(_):
    pass


def test_text_run_end_to_end_and_report_metrics_match_saved_predictions(text_cfg, text_csv):
    res = run_experiment(text_cfg, data_path=text_csv, run_id="t1", log=_quiet)
    rep = json.loads((res.run_dir / "report.json").read_text())
    assert rep["task"] == "text" and rep["split"]["sizes"]["train"] > rep["split"]["sizes"]["test"] > 0
    labels = rep["data"]["labels"]
    for m in rep["models"]:
        for split in ("val", "test"):
            p = pd.read_csv(res.run_dir / "predictions" / f"{m['name']}_{split}.csv", dtype=str)
            recomputed = compute_metrics(p["y_true"], p["y_pred"], labels)
            assert recomputed["accuracy"] == pytest.approx(m[split]["metrics"]["accuracy"])
            assert recomputed["macro_f1"] == pytest.approx(m[split]["metrics"]["macro_f1"])
    for f in rep["figures"]:
        assert (res.run_dir / f).stat().st_size > 1000
    assert (res.run_dir / "summary.md").exists() and (res.run_dir / "error_analysis.json").exists()
    assert rep["environment"]["scikit-learn"] and rep["config_hash"] and rep["data"]["sha256"]


def test_selection_uses_validation_not_test(text_cfg, text_csv):
    rep = run_experiment(text_cfg, data_path=text_csv, run_id="t2", log=_quiet).report
    val = {m["name"]: m["val"]["metrics"]["macro_f1"] for m in rep["models"]}
    best = max(val.values())
    first_best = next(n for n in (m["name"] for m in rep["models"]) if val[n] == best)
    assert rep["selected_model"] == first_best
    assert "test split plays no role" in rep["selection_rule"]


def test_runs_are_reproducible(text_cfg, text_csv):
    a = run_experiment(text_cfg, data_path=text_csv, run_id="ra", log=_quiet).report
    b = run_experiment(text_cfg, data_path=text_csv, run_id="rb", log=_quiet).report
    for ma, mb in zip(a["models"], b["models"]):
        assert ma["test"]["metrics"]["accuracy"] == mb["test"]["metrics"]["accuracy"]
        assert ma["best_params"] == mb["best_params"]
    sa = pd.read_csv(a_path(text_cfg, "ra"))
    sb = pd.read_csv(a_path(text_cfg, "rb"))
    assert sa.equals(sb)


def a_path(cfg, run_id):
    from pathlib import Path
    return Path(cfg.output.reports_dir) / run_id / "split_assignments.csv.gz"


def test_tabular_run_threshold_chosen_on_validation(tab_cfg, tab_csv):
    res = run_experiment(tab_cfg, data_path=tab_csv, run_id="tab", log=_quiet)
    rep = res.report
    assert rep["split"]["strategy"] == "predefined" and rep["data"]["positive_label"] == "1"
    lr = next(m for m in rep["models"] if m["name"] == "logreg")
    val = pd.read_csv(res.run_dir / "predictions" / "logreg_val.csv", dtype={"y_true": str, "y_pred": str})
    thr, _ = select_threshold(val["y_true"], val["score"], "1", "f1_positive", default=0.5)
    assert lr["threshold"]["value"] == pytest.approx(thr)
    assert "all_positive" in rep["baselines"]


def test_metadata_columns_never_become_features(tab_cfg, tmp_path):
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=4)
    d["PHQ_Score"] = np.where(d["label"] == "1", 15, 3)        # perfectly predictive leak, if it were used
    d["Gender"] = "male"
    p = tmp_path / "with_meta.csv"
    d.to_csv(p, index=False)
    tab_cfg.data.exclude_columns = ["PHQ_Score", "Gender"]
    res = run_experiment(tab_cfg, data_path=p, run_id="meta", log=_quiet)
    cols = load_bundle(res.model_path).feature_columns
    assert "PHQ_Score" not in cols and "Gender" not in cols and "label" not in cols and "split" not in cols
    assert res.report["models"][1]["test"]["metrics"]["accuracy"] < 0.99


def test_run_writes_loadable_model_with_same_predictions(text_cfg, text_csv):
    res = run_experiment(text_cfg, data_path=text_csv, run_id="persist", log=_quiet)
    bundle = load_bundle(res.model_path)
    te = pd.read_csv(res.run_dir / "predictions" / f"{res.report['selected_model']}_test.csv", dtype=str)
    texts = pd.read_csv(text_csv, dtype=str).set_index("id").loc[te["record_id"], "text"].tolist()
    assert list(bundle.predict_frame(texts)["prediction"]) == list(te["y_pred"])


def test_invalid_data_stops_training(text_cfg, tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["1", "2", "3", "4"], "text": ["aa bb", "cc dd", "ee ff", "gg hh"], "label": ["anxiety", "nonsense", "anxiety", "lonely"]}).to_csv(p, index=False)
    with pytest.raises(DataError, match="validation failed"):
        run_experiment(text_cfg, data_path=p, log=_quiet)


def test_missing_data_file(text_cfg, tmp_path):
    with pytest.raises(DataError, match="not found"):
        run_experiment(text_cfg, data_path=tmp_path / "absent.csv", log=_quiet)
