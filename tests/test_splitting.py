import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mindtext.config import DataConfig, SplitConfig
from mindtext.data import validate_and_clean
from mindtext.errors import DataError, LeakageError
from mindtext.models import build_pipeline
from mindtext.splitting import SplitResult, leakage_report, make_split
from mindtext.synthetic import make_tabular_dataset


@pytest.fixture
def clean(text_df):
    cfg = DataConfig(task="text", id_column="id", group_column="group", time_column="time")
    df, _ = validate_and_clean(text_df, cfg)
    return df, cfg


def test_split_is_reproducible_and_seed_sensitive(clean):
    df, _ = clean
    a, b = make_split(df, SplitConfig(), 7), make_split(df, SplitConfig(), 7)
    c = make_split(df, SplitConfig(), 8)
    for n in ("train", "val", "test"):
        assert np.array_equal(a.indices(n), b.indices(n))
    assert not np.array_equal(a.test_idx, c.test_idx)


def test_partitions_are_disjoint_and_cover_everything(clean):
    df, cfg = clean
    s = make_split(df, SplitConfig(val_size=0.15, test_size=0.2), 0)
    allidx = np.concatenate([s.train_idx, s.val_idx, s.test_idx])
    assert len(allidx) == len(set(allidx)) == len(df)
    assert abs(len(s.test_idx) / len(df) - 0.2) < 0.03 and abs(len(s.val_idx) / len(df) - 0.15) < 0.03
    assert leakage_report(df, s, cfg)["index_overlap"] == {"train|val": 0, "train|test": 0, "val|test": 0}


def test_stratification_keeps_every_class_in_every_split(clean):
    df, _ = clean
    s = make_split(df, SplitConfig(), 0)
    for name, counts in s.class_counts(df).items():
        assert set(counts) == set(df["label"]), name


def test_grouped_split_has_no_group_overlap(clean):
    df, cfg = clean
    s = make_split(df, SplitConfig(strategy="grouped", val_size=0.15, test_size=0.2), 0)
    rep = leakage_report(df, s, cfg)
    assert rep["group_overlap"] == {"train|val": 0, "train|test": 0, "val|test": 0}
    assert len(s.test_idx) > 0 and len(s.train_idx) > len(s.test_idx)


def test_grouped_overlap_is_detected_when_forced(clean):
    df, cfg = clean
    bad = SplitResult(np.array([0, 1]), np.array([2]), np.array([3]), "grouped", 0)
    df2 = df.copy()
    df2["group"] = "same"
    with pytest.raises(LeakageError, match="shares groups"):
        leakage_report(df2, bad, cfg)


def test_stratified_split_reports_group_overlap_as_information(clean):
    df, cfg = clean
    rep = leakage_report(df, make_split(df, SplitConfig(), 0), cfg)
    assert sum(rep["group_overlap"].values()) > 0 and rep["clean"] is False


def test_chronological_split_orders_by_time(clean):
    df, _ = clean
    s = make_split(df, SplitConfig(strategy="chronological", val_size=0.1, test_size=0.2), 0)
    t = pd.to_datetime(df["time"])
    assert t.iloc[s.train_idx].max() <= t.iloc[s.val_idx].min()
    assert t.iloc[s.val_idx].max() <= t.iloc[s.test_idx].min()


def test_chronological_rejects_unparsable_time(clean):
    df, _ = clean
    df = df.copy()
    df.loc[0, "time"] = "not a date"
    with pytest.raises(DataError, match="unparsable"):
        make_split(df, SplitConfig(strategy="chronological"), 0)


def test_predefined_split_and_unknown_value():
    raw = make_tabular_dataset(seed=3)
    cfg = DataConfig(task="tabular", id_column="participant_id", split_column="split", positive_label="1")
    df, _ = validate_and_clean(raw, cfg)
    s = make_split(df, SplitConfig(strategy="predefined", split_names={"train": ["train"], "val": ["dev"], "test": ["test"]}), 0)
    assert s.sizes() == {"train": 90, "val": 30, "test": 30}
    df2 = df.copy()
    df2.loc[0, "split"] = "holdout"
    with pytest.raises(DataError, match="Unrecognised"):
        make_split(df2, SplitConfig(strategy="predefined"), 0)


def test_tiny_class_gives_clear_error():
    df = pd.DataFrame({"record_id": [f"r{i}" for i in range(12)], "label": ["a"] * 11 + ["b"], "text": [f"t{i}" for i in range(12)]})
    with pytest.raises(DataError, match="Could not create"):
        make_split(df, SplitConfig(), 0)


def test_cross_split_duplicate_texts_are_flagged(clean):
    df, cfg = clean
    df = df.copy()
    s = make_split(df, SplitConfig(), 0)
    df.loc[s.test_idx[0], "text"] = df.loc[s.train_idx[0], "text"]
    assert leakage_report(df, s, cfg)["cross_split_duplicates"]["train|test"] >= 1


def test_vectorizer_vocabulary_is_learned_from_training_split_only(text_cfg, text_df):
    """A token that only exists in a held-out record must not enter the TF-IDF vocabulary."""
    df, _ = validate_and_clean(text_df, DataConfig(task="text", id_column="id"))
    s = make_split(df, SplitConfig(), 0)
    df = df.copy()
    df.loc[s.test_idx[0], "text"] = df.loc[s.test_idx[0], "text"] + " zzheldoutonly"
    df.loc[s.train_idx[0], "text"] = df.loc[s.train_idx[0], "text"] + " zztrainonly"
    pipe: Pipeline = build_pipeline(text_cfg, text_cfg.models[1])
    pipe.fit(df["text"].iloc[s.train_idx].tolist(), df["label"].iloc[s.train_idx])
    vocab = pipe.named_steps["tfidf"].vocabulary_
    assert "zztrainonly" in vocab and "zzheldoutonly" not in vocab
    pipe.predict(df["text"].iloc[s.test_idx].tolist())  # transform-only on held-out data
    assert "zzheldoutonly" not in pipe.named_steps["tfidf"].vocabulary_


def test_imputer_and_scaler_statistics_come_from_training_rows_only(tab_cfg):
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y_train = np.array(["0", "1"] * 20)
    pipe = build_pipeline(tab_cfg, tab_cfg.models[1])
    pipe.fit(X_train, y_train)
    X_test = pd.DataFrame(np.full((5, 3), 1000.0), columns=list("abc"))
    pipe.predict(X_test)
    assert np.allclose(pipe.named_steps["scale"].mean_, X_train.mean().to_numpy())
    assert np.allclose(pipe.named_steps["impute"].statistics_, X_train.median().to_numpy())
