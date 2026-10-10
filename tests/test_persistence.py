import json

import pytest

from mindtext.errors import IntegrityError, ModelError
from mindtext.models import ModelBundle, build_pipeline
from mindtext.persistence import load_bundle, meta_path, save_bundle


@pytest.fixture
def bundle(text_cfg, text_df):
    pipe = build_pipeline(text_cfg, text_cfg.models[1]).fit(text_df["text"], text_df["label"])
    return ModelBundle(pipe, "text", sorted(text_df["label"].unique()), text_column="text", metadata={"k": 1})


def test_roundtrip_gives_identical_predictions(tmp_path, bundle, text_df):
    p = save_bundle(bundle, tmp_path / "sub" / "m.joblib")
    loaded = load_bundle(p)
    texts = text_df["text"].head(15).tolist()
    assert list(loaded.predict_frame(texts)["prediction"]) == list(bundle.predict_frame(texts)["prediction"])
    assert loaded.labels == bundle.labels and loaded.metadata == {"k": 1}


def test_metadata_sidecar_contents(tmp_path, bundle):
    p = save_bundle(bundle, tmp_path / "m.joblib", {"run_id": "abc"})
    meta = json.loads(meta_path(p).read_text())
    assert len(meta["sha256"]) == 64 and meta["run_id"] == "abc" and meta["task"] == "text"
    assert meta["environment"]["scikit-learn"] and "trusted" in meta["security_note"]


def test_tampered_file_is_rejected(tmp_path, bundle):
    p = save_bundle(bundle, tmp_path / "m.joblib")
    with open(p, "ab") as fh:
        fh.write(b"corruption")
    with pytest.raises(IntegrityError, match="Checksum mismatch"):
        load_bundle(p)


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelError, match="not found"):
        load_bundle(tmp_path / "ghost.joblib")


def test_missing_sidecar_warns(tmp_path, bundle):
    p = save_bundle(bundle, tmp_path / "m.joblib")
    meta_path(p).unlink()
    with pytest.warns(UserWarning, match="integrity cannot be verified"):
        load_bundle(p)


def test_wrong_object_type_rejected(tmp_path):
    import joblib
    p = tmp_path / "x.joblib"
    joblib.dump({"not": "a bundle"}, p)
    with pytest.raises(ModelError, match="not contain a MindText model bundle"):
        load_bundle(p)
