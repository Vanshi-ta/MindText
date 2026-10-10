import pytest

from mindtext.config import Config
from mindtext.synthetic import make_tabular_dataset, make_text_dataset


@pytest.fixture
def text_df():
    return make_text_dataset(n_per_class=30, seed=1)


@pytest.fixture
def text_csv(tmp_path, text_df):
    p = tmp_path / "text.csv"
    text_df[["id", "text", "label"]].to_csv(p, index=False)
    return p


@pytest.fixture
def tab_csv(tmp_path):
    p = tmp_path / "tab.csv"
    make_tabular_dataset(seed=2).to_csv(p, index=False)
    return p


def _models_text():
    return [
        {"name": "majority", "type": "majority"},
        {"name": "logreg", "type": "logreg", "tune": {"C": [0.5, 2.0]}},
        {"name": "nb", "type": "nb"},
    ]


@pytest.fixture
def text_cfg(tmp_path):
    return Config.from_dict({
        "data": {"path": "unused.csv", "task": "text", "id_column": "id",
                 "allowed_labels": ["anxiety", "depression", "lonely", "mentalhealth", "suicidewatch"]},
        "split": {"strategy": "stratified", "val_size": 0.15, "test_size": 0.2},
        "tfidf": {"ngram_range": [1, 1], "min_df": 1, "max_features": 500},
        "models": _models_text(),
        "train": {"cv_folds": 3, "bootstrap_samples": 50},
        "output": {"reports_dir": str(tmp_path / "reports"), "artifacts_dir": str(tmp_path / "artifacts")},
    })


@pytest.fixture
def tab_cfg(tmp_path):
    return Config.from_dict({
        "data": {"path": "unused.csv", "task": "tabular", "id_column": "participant_id", "split_column": "split",
                 "positive_label": "1", "allowed_labels": ["0", "1"]},
        "split": {"strategy": "predefined"},
        "models": [
            {"name": "majority", "type": "majority"},
            {"name": "logreg", "type": "logreg", "params": {"class_weight": "balanced"}, "tune": {"C": [0.1, 1.0]}},
            {"name": "svm", "type": "svm_rbf", "params": {"class_weight": "balanced"}},
            {"name": "rf", "type": "rf", "params": {"n_estimators": 30}},
        ],
        "threshold": {"enabled": True},
        "train": {"cv_folds": 3, "bootstrap_samples": 50, "tune_metric": "f1_positive"},
        "output": {"reports_dir": str(tmp_path / "reports"), "artifacts_dir": str(tmp_path / "artifacts")},
    })
