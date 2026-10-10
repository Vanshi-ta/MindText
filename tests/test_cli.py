import json

import pandas as pd
import pytest

from mindtext.cli import main


@pytest.fixture
def trained(tmp_path, text_csv, text_cfg):
    import yaml
    cfgd = text_cfg.to_dict()
    cfgd["data"]["path"] = str(text_csv)
    cfgd["output"] = {"reports_dir": str(tmp_path / "reports"), "artifacts_dir": str(tmp_path / "artifacts"), "run_name": None}
    cp = tmp_path / "cfg.yaml"
    cp.write_text(yaml.safe_dump(cfgd))
    return cp, tmp_path


def test_make_sample_then_data_check(tmp_path, capsys):
    out = tmp_path / "s.csv"
    assert main(["make-sample", "--task", "text", "--out", str(out), "--n", "12"]) == 0
    assert main(["data-check", "--data", str(out), "--out", str(tmp_path / "q.json")]) == 0
    assert "class distribution" in capsys.readouterr().out
    assert json.loads((tmp_path / "q.json").read_text())["rows_kept"] == 60


def test_data_check_returns_1_on_errors(tmp_path):
    p = tmp_path / "one.csv"
    pd.DataFrame({"text": ["a b", "c d"], "label": ["x", "x"]}).to_csv(p, index=False)
    assert main(["data-check", "--data", str(p)]) == 1


def test_missing_file_is_a_clean_error(tmp_path, capsys):
    assert main(["data-check", "--data", str(tmp_path / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["evaluate", "--model", str(tmp_path / "m.joblib"), "--data", str(tmp_path / "d.csv")]) == 2
    assert main(["train", "--config", str(tmp_path / "missing.yaml")]) == 2


def test_argument_validation():
    with pytest.raises(SystemExit) as e:
        main(["train"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        main(["not-a-command"])
    with pytest.raises(SystemExit):
        main(["report"])  # needs --latest/--run-id/--path


def test_train_report_predict_evaluate_flow(trained, text_csv, capsys):
    cfg, tmp = trained
    assert main(["train", "--config", str(cfg)]) == 0
    assert "NOT a diagnosis" in capsys.readouterr().out
    model = tmp / "artifacts" / "model.joblib"
    assert model.exists()
    assert main(["report", "--latest", "--reports-dir", str(tmp / "reports")]) == 0
    assert "selected on the validation split" in capsys.readouterr().out
    assert main(["predict", "--model", str(model), "--text", "worry about the exam"]) == 0
    out = capsys.readouterr().out
    assert "predicted class" in out and "NOT a diagnosis" in out
    assert main(["predict", "--model", str(model), "--text", "tired and empty", "--json"]) == 0
    assert main(["evaluate", "--model", str(model), "--data", str(text_csv), "--out", str(tmp / "m.json")]) == 0
    assert "macro_f1" in capsys.readouterr().out and (tmp / "m.json").exists()
    assert main(["predict", "--model", str(model)]) == 2   # no text given


def test_report_with_no_runs(tmp_path, capsys):
    assert main(["report", "--latest", "--reports-dir", str(tmp_path / "empty")]) == 2
    assert "No experiment reports" in capsys.readouterr().err


def test_evaluate_rejects_unknown_labels(trained, tmp_path):
    cfg, tmp = trained
    main(["train", "--config", str(cfg)])
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["aa bb", "cc dd"], "label": ["anxiety", "martian"]}).to_csv(bad, index=False)
    assert main(["evaluate", "--model", str(tmp / "artifacts" / "model.joblib"), "--data", str(bad)]) == 2
