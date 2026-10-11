import importlib.util

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mindtext.config import ModelSpec
from mindtext.errors import MissingDependencyError, ModelError
from mindtext.models import (BalancedSampleWeightClassifier, LabelEncodedClassifier, ModelBundle, build_pipeline, positive_scores,
                             predict_with_scores)

HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_LGBM = importlib.util.find_spec("lightgbm") is not None


@pytest.mark.parametrize("mtype", ["majority", "nb", "logreg", "linearsvc", "rf"])
def test_text_pipelines_build_fit_predict(text_cfg, text_df, mtype):
    spec = ModelSpec(name=mtype, type=mtype, params={"n_estimators": 10} if mtype == "rf" else {})
    pipe = build_pipeline(text_cfg, spec)
    assert isinstance(pipe, Pipeline) and [n for n, _ in pipe.steps] == ["clean", "tfidf", "clf"]
    pipe.fit(text_df["text"], text_df["label"])
    pred = pipe.predict(text_df["text"].head(7))
    assert pred.shape == (7,) and set(pred) <= set(text_df["label"])


@pytest.mark.parametrize("mtype", ["majority", "logreg", "linearsvc", "svm_rbf", "rf"])
def test_tabular_pipelines_build_fit_predict(tab_cfg, mtype):
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    X, y = d.filter(like="_mean"), d["label"]
    pipe = build_pipeline(tab_cfg, ModelSpec(name=mtype, type=mtype, params={"n_estimators": 10} if mtype == "rf" else {}))
    assert pipe.steps[0][0] == "impute"
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


def test_nb_rejected_for_tabular(tab_cfg):
    with pytest.raises(ModelError, match="Naive Bayes"):
        build_pipeline(tab_cfg, ModelSpec(name="nb", type="nb"))


def test_missing_values_are_handled_by_imputer(tab_cfg):
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    X, y = d.filter(like="_mean").copy(), d["label"]
    X.iloc[::7, 0] = np.nan
    X["all_nan"] = np.nan
    pipe = build_pipeline(tab_cfg, ModelSpec(name="lr", type="logreg")).fit(X, y)
    assert np.isfinite(pipe.predict_proba(X)).all()


@pytest.mark.skipif(HAS_XGB, reason="xgboost installed")
def test_xgboost_without_package_has_clear_error(tab_cfg):
    with pytest.raises(MissingDependencyError, match="boosting"):
        build_pipeline(tab_cfg, ModelSpec(name="x", type="xgboost"))


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
def test_xgboost_pipeline_with_string_labels(tab_cfg):
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    spec = ModelSpec(name="x", type="xgboost", params={"n_estimators": 5, "scale_pos_weight": "auto"})
    pipe = build_pipeline(tab_cfg, spec).fit(d.filter(like="_mean"), d["label"])
    assert set(pipe.predict(d.filter(like="_mean"))) <= {"0", "1"}


@pytest.mark.skipif(not HAS_LGBM, reason="lightgbm not installed")
def test_lightgbm_pipeline(text_cfg, text_df):
    pipe = build_pipeline(text_cfg, ModelSpec(name="l", type="lightgbm", params={"n_estimators": 5, "min_child_samples": 2}))
    pipe.fit(text_df["text"], text_df["label"])
    assert pipe.predict(text_df["text"].head(3)).shape == (3,)


def test_label_encoded_wrapper_and_auto_scale_pos_weight():
    class Probe(LogisticRegression):
        def __init__(self, scale_pos_weight=None, **kw):
            super().__init__(**kw)
            self.scale_pos_weight = scale_pos_weight

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.array(["no"] * 30 + ["yes"] * 10)
    w = LabelEncodedClassifier(Probe(scale_pos_weight="auto")).fit(X, y)
    assert list(w.classes_) == ["no", "yes"]
    assert w.estimator_.scale_pos_weight == pytest.approx(3.0)
    assert set(w.predict(X)) <= {"no", "yes"} and w.predict_proba(X).shape == (40, 2)


def test_svm_uses_margin_not_platt_probabilities(tab_cfg):
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    pipe = build_pipeline(tab_cfg, ModelSpec(name="s", type="svm_rbf")).fit(d.filter(like="_mean"), d["label"])
    assert not hasattr(pipe, "predict_proba")
    s, kind = positive_scores(pipe, d.filter(like="_mean"), "1")
    assert kind == "decision" and s.shape == (len(d),)


def test_threshold_changes_predictions_monotonically(tab_cfg):
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    X = d.filter(like="_mean")
    pipe = build_pipeline(tab_cfg, ModelSpec(name="l", type="logreg")).fit(X, d["label"])
    low = (predict_with_scores(pipe, X, "1", 0.1)[0] == "1").sum()
    high = (predict_with_scores(pipe, X, "1", 0.9)[0] == "1").sum()
    assert low >= high and low > high


def test_bundle_predict_frame_shapes_and_validation(tab_cfg, text_cfg, text_df):
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    cols = [c for c in d.columns if c.endswith("_mean")]
    pipe = build_pipeline(tab_cfg, ModelSpec(name="l", type="logreg")).fit(d[cols], d["label"])
    b = ModelBundle(pipe, "tabular", ["0", "1"], "1", 0.4, feature_columns=cols)
    out = b.predict_frame(d.head(6))
    assert out.shape[0] == 6 and {"prediction", "score", "proba_0", "proba_1"} <= set(out.columns)
    with pytest.raises(ModelError, match="missing"):
        b.predict_frame(d[cols[:-2]].head(3))
    with pytest.raises(ModelError):
        b.predict_frame(["text"])
    tp = build_pipeline(text_cfg, text_cfg.models[1]).fit(text_df["text"], text_df["label"])
    tb = ModelBundle(tp, "text", sorted(text_df["label"].unique()), text_column="text")
    assert tb.predict_frame(["worry about the exam", "tired and empty"]).shape[0] == 2
    with pytest.raises(ModelError, match="No input"):
        tb.predict_frame([])


def test_rf_balanced_uses_explicit_sample_weights_not_class_weight_param(tab_cfg):
    from sklearn.ensemble import RandomForestClassifier
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    X, y = d.filter(like="_mean"), d["label"]
    spec = ModelSpec(name="rf", type="rf", params={"n_estimators": 20, "class_weight": "balanced"})
    pipe = build_pipeline(tab_cfg, spec)
    clf = pipe.named_steps["clf"]
    assert isinstance(clf, BalancedSampleWeightClassifier)
    assert clf.estimator.class_weight is None            # sklearn's own class_weight path is bypassed
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),) and set(pipe.classes_) == {"0", "1"}
    assert pipe.predict_proba(X).shape == (len(X), 2)
    assert clf.feature_importances_.shape[0] == X.shape[1]


def test_balanced_weights_equal_class_totals_and_match_sklearn_balanced():
    from sklearn.utils.class_weight import compute_sample_weight

    seen = {}

    class Spy(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            seen["w"] = sample_weight
            return super().fit(X, y, sample_weight=sample_weight)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = np.array(["a"] * 40 + ["b"] * 10)
    BalancedSampleWeightClassifier(Spy()).fit(X, y)
    w = seen["w"]
    assert w[y == "a"].sum() == pytest.approx(w[y == "b"].sum())
    assert np.allclose(w, compute_sample_weight("balanced", y))


def test_svm_does_not_pass_deprecated_probability_param(tab_cfg):
    spec = ModelSpec(name="s", type="svm_rbf")
    svc = build_pipeline(tab_cfg, spec).named_steps["clf"]
    import warnings
    from mindtext.synthetic import make_tabular_dataset
    d = make_tabular_dataset(seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        build_pipeline(tab_cfg, spec).fit(d.filter(like="_mean"), d["label"])
    assert svc.probability is False
