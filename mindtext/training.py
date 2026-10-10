"""Experiment runner: load -> validate -> split -> (tune on train) -> select on
validation -> evaluate once on test -> report.

Leakage controls (each is covered by a test):
  * all learned preprocessing (TF-IDF vocabulary/IDF, imputation, scaling,
    variance/feature selection) lives inside the Pipeline and is fitted on the
    training split (or on the training folds during cross-validation);
  * hyper-parameters are tuned by cross-validation on the training split only;
  * model selection and the optional decision threshold use the validation split;
  * the test split is evaluated after those choices are frozen, and never used
    to choose anything.
"""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from . import DISCLAIMER
from .config import Config
from .data import load_table, resolve_positive_label, save_quality_report, validate_and_clean
from .errors import ConfigError, DataError
from .error_analysis import collect_errors, top_features
from .evaluation import bootstrap_ci, compute_metrics, metric_value, select_threshold, trivial_baselines
from .models import (
    ModelBundle,
    build_pipeline,
    default_threshold,
    feature_names_after_selection,
    positive_scores,
    predict_with_scores,
)
from .persistence import environment_info, save_bundle
from .reporting import make_figures, render_markdown_summary, write_json
from .splitting import SplitResult, leakage_report, make_split

SELECTION_RULE = "highest validation {metric}; ties resolved by order in the config; the test split plays no role"


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    model_path: Path
    report: dict


def make_run_id(cfg: Config, now: datetime | None = None) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    name = f"_{cfg.output.run_name}" if cfg.output.run_name else ""
    return f"{ts}_{cfg.data.task}_{cfg.config_hash()[:6]}{name}"


def _scorer(name: str, positive: str | None):
    if name == "f1_positive":
        if positive is None:
            raise ConfigError("tune_metric 'f1_positive' needs a binary task with data.positive_label.")
        return make_scorer(f1_score, pos_label=positive, zero_division=0)
    return name


def _cv(train_labels: np.ndarray, folds: int, seed: int, groups: np.ndarray | None, notes: list[str]):
    min_count = int(pd.Series(train_labels).value_counts().min())
    k = min(folds, min_count)
    if k < 2:
        notes.append("Hyper-parameter tuning skipped: smallest training class has fewer than 2 records.")
        return None
    if k < folds:
        notes.append(f"cv_folds reduced from {folds} to {k} (smallest training class has {min_count} records).")
    if groups is not None:
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def _inputs(df: pd.DataFrame, cfg: Config, feature_cols: list[str], idx: np.ndarray):
    part = df.iloc[idx]
    if cfg.data.task == "text":
        return part["text"].tolist()
    return part[feature_cols].reset_index(drop=True)


def _pred_frame(df, idx, split_name, pred, score, kind) -> pd.DataFrame:
    part = df.iloc[idx]
    return pd.DataFrame(
        {
            "record_id": part["record_id"].to_numpy(),
            "split": split_name,
            "y_true": part["label"].to_numpy(),
            "y_pred": pred,
            "score": score if score is not None else np.nan,
            "score_kind": kind,
        }
    )


def run_experiment(
    cfg: Config,
    data_path: str | Path | None = None,
    run_id: str | None = None,
    log: Callable[[str], None] = print,
) -> RunResult:
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    notes: list[str] = []

    # ---- 1. data ------------------------------------------------------
    path = Path(data_path or cfg.data.path)
    log(f"[1/6] Loading {path}")
    raw = load_table(path, cfg.data)
    df, quality = validate_and_clean(raw, cfg.data, source_path=path)
    if quality.has_errors:
        raise DataError("Data validation failed:\n" + quality.format_text())
    labels = sorted(df["label"].unique())
    positive = resolve_positive_label(labels, cfg.data.positive_label)
    feature_cols = quality.feature_columns
    log(f"      {len(df)} usable records, {len(labels)} classes, {len(feature_cols)} feature columns")
    for w in quality.warnings:
        notes.append(f"data: {w['message']}")

    # ---- 2. split + leakage checks -------------------------------------
    log(f"[2/6] Splitting ({cfg.split.strategy}, seed={seed})")
    split: SplitResult = make_split(df, cfg.split, seed)
    leak = leakage_report(df, split, cfg.data, feature_cols if cfg.data.task == "tabular" else None)
    if any(leak["cross_split_duplicates"].values()):
        notes.append(f"{sum(leak['cross_split_duplicates'].values())} duplicated text/feature record(s) occur in more than one split {leak['cross_split_duplicates']}; scores may be optimistic.")
    if leak["group_overlap"] and any(leak["group_overlap"].values()):
        notes.append(f"Groups (e.g. authors) are shared across splits {leak['group_overlap']}; consider split.strategy='grouped'.")
    if len(split.val_idx) == 0:
        notes.append("No validation split: model selection/threshold selection unavailable.")
    if split.strategy == "stratified" and cfg.data.task == "text" and not cfg.data.group_column:
        notes.append("No group/author column: posts by the same person may fall in different splits (cannot be checked).")
    tr, va, te = split.train_idx, split.val_idx, split.test_idx
    X_tr, y_tr = _inputs(df, cfg, feature_cols, tr), df["label"].iloc[tr].to_numpy()
    groups_tr = df["group"].iloc[tr].astype(str).to_numpy() if (cfg.split.strategy == "grouped" and "group" in df) else None

    # ---- 3. fit models --------------------------------------------------
    run_id = run_id or make_run_id(cfg)
    run_dir = Path(cfg.output.reports_dir) / run_id
    art_dir = Path(cfg.output.artifacts_dir) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    save_quality_report(quality, run_dir / "data_quality.json")
    split.assignment(df).to_csv(run_dir / "split_assignments.csv.gz", index=False)

    results: list[dict] = []
    fitted: dict[str, object] = {}
    thresholds: dict[str, float | None] = {}
    pred_frames: dict[str, pd.DataFrame] = {}
    log(f"[3/6] Training {len(cfg.models)} model(s)")
    for spec in cfg.models:
        t0 = time.perf_counter()
        pipe = build_pipeline(cfg, spec)
        best_params, cv_score = None, None
        if spec.tune:
            grid = {(k if "__" in k else f"clf__{k}"): list(v) for k, v in spec.tune.items()}
            cv = _cv(y_tr, cfg.train.cv_folds, seed, groups_tr, notes)
            if cv is not None:
                search = GridSearchCV(pipe, grid, scoring=_scorer(cfg.train.tune_metric, positive), cv=cv,
                                      n_jobs=cfg.train.n_jobs, refit=True, error_score="raise")
                search.fit(X_tr, y_tr, groups=groups_tr)
                pipe, best_params, cv_score = search.best_estimator_, search.best_params_, float(search.best_score_)
            else:
                pipe.fit(X_tr, y_tr)
        else:
            pipe.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - t0
        fitted[spec.name] = pipe

        # decision threshold: validation split only
        thr: float | None = None
        thr_info = None
        if positive is not None and cfg.threshold.enabled:
            s_val, kind = positive_scores(pipe, _inputs(df, cfg, feature_cols, va), positive)
            thr, best = select_threshold(df["label"].iloc[va].to_numpy(), s_val, positive, cfg.threshold.metric,
                                         default_threshold(kind), cfg.threshold.max_candidates)
            pos_rate = float((s_val >= thr).mean())
            thr_info = {"value": thr, "metric": cfg.threshold.metric, "validation_value": best, "score_kind": kind,
                        "validation_predicted_positive_rate": pos_rate}
            if pos_rate >= 0.95:
                notes.append(f"{spec.name}: the validation-selected threshold labels {pos_rate:.0%} of validation records positive "
                             "(near all-positive rule); compare with the all-positive baseline before trusting its F1.")
        thresholds[spec.name] = thr

        entry = {
            "name": spec.name, "type": spec.type, "provenance": spec.provenance, "params": spec.params,
            "best_params": best_params, "cv_score": cv_score, "cv_metric": cfg.train.tune_metric if best_params else None,
            "threshold": thr_info, "fit_seconds": round(fit_seconds, 3), "val": None, "test": None, "test_ci": None,
        }
        for split_name, idx in (("val", va), ("test", te)):
            if len(idx) == 0:
                continue
            pred, score, kind = predict_with_scores(pipe, _inputs(df, cfg, feature_cols, idx), positive, thr)
            y_true = df["label"].iloc[idx].to_numpy()
            m = compute_metrics(y_true, pred, labels, y_score=score if positive is not None else None, positive_label=positive)
            entry[split_name] = {"metrics": m, "score_kind": kind}
            frame = _pred_frame(df, idx, split_name, pred, score, kind)
            pred_frames[f"{spec.name}|{split_name}"] = frame
            frame.to_csv(run_dir / "predictions" / f"{spec.name}_{split_name}.csv", index=False)
            if split_name == "test" and cfg.train.bootstrap_samples > 0:
                entry["test_ci"] = bootstrap_ci(y_true, pred, labels, positive, cfg.train.bootstrap_samples, seed)
        results.append(entry)
        log(f"      {spec.name:<20} fit {fit_seconds:6.1f}s")

    # ---- 4. selection (validation only) ---------------------------------
    metric = cfg.train.selection_metric
    if len(results) == 1:
        selected = results[0]["name"]
        rule = "single model configured; nothing to select"
    else:
        if any(r["val"] is None for r in results):
            raise ConfigError("Model selection requires a validation split.")
        scores = [metric_value(r["val"]["metrics"], metric) for r in results]
        selected = results[int(np.argmax(scores))]["name"]  # argmax returns the first maximum
        rule = SELECTION_RULE.format(metric=metric)
    log(f"[4/6] Selected '{selected}' by validation {metric}")

    # ---- 5. error analysis, model, artefacts ----------------------------
    sel_pipe, sel_thr = fitted[selected], thresholds[selected]
    sel_frame = pred_frames[f"{selected}|test"]
    texts = dict(zip(df["record_id"].iloc[te], df["text"].iloc[te])) if cfg.data.task == "text" else None
    ea = collect_errors(sel_frame, positive, texts, cfg.error_analysis.n_examples,
                        cfg.error_analysis.include_text and cfg.data.task == "text", cfg.error_analysis.excerpt_chars)
    names = feature_names_after_selection(sel_pipe, feature_cols) if cfg.data.task == "tabular" else None
    ea["top_features"] = top_features(sel_pipe, cfg.error_analysis.top_features, names)
    write_json(ea, run_dir / "error_analysis.json")

    bundle = ModelBundle(
        pipeline=sel_pipe, task=cfg.data.task, labels=labels, positive_label=positive, threshold=sel_thr,
        text_column="text" if cfg.data.task == "text" else None,
        feature_columns=feature_cols if cfg.data.task == "tabular" else None,
        metadata={"run_id": run_id, "model_name": selected, "config_hash": cfg.config_hash(), "seed": seed},
    )
    model_path = save_bundle(bundle, art_dir / "model.joblib", {"run_id": run_id, "data_sha256": quality.file_sha256})
    if cfg.train.save_all_models:
        for n, p in fitted.items():
            if n != selected:
                save_bundle(ModelBundle(p, cfg.data.task, labels, positive, thresholds[n], bundle.text_column, bundle.feature_columns,
                                        {"run_id": run_id, "model_name": n}), art_dir / f"{n}.joblib")
    latest = Path(cfg.output.artifacts_dir) / "model.joblib"
    shutil.copyfile(model_path, latest)
    shutil.copyfile(model_path.with_name(model_path.name + ".meta.json"), latest.with_name(latest.name + ".meta.json"))

    # ---- 6. report ------------------------------------------------------
    y_test = df["label"].iloc[te].to_numpy()
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": cfg.data.task,
        "disclaimer": DISCLAIMER,
        "environment": environment_info(),
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "data": {
            "path": str(path), "sha256": quality.file_sha256, "description": cfg.data.description,
            "labels": labels, "positive_label": positive, "n_records": len(df),
            "class_distribution": quality.class_distribution, "feature_columns": len(feature_cols),
            "quality_summary": {"rows_read": quality.rows_read, "rows_kept": quality.rows_kept, "counts": quality.counts,
                                "n_errors": len(quality.errors), "n_warnings": len(quality.warnings)},
        },
        "split": {"strategy": split.strategy, "seed": seed, "sizes": split.sizes(), "class_counts": split.class_counts(df), "leakage_checks": leak},
        "models": results,
        "selected_model": selected,
        "selection_rule": rule,
        "baselines": {"split": "test", **trivial_baselines(y_test, labels, positive)},
        "error_analysis_file": "error_analysis.json",
        "model_path": str(model_path),
        "warnings": notes,
        "evaluation_protocol": (
            "Fit/tune on train (CV inside train); choose model (and threshold, if enabled) on validation; "
            "report test metrics for every model after selection. Test metrics must not be used to pick a model."
        ),
    }
    figures = make_figures(report, pred_frames, run_dir)
    report["figures"] = [str(Path(f).relative_to(run_dir)) for f in figures]
    write_json(report, run_dir / "report.json")
    (run_dir / "summary.md").write_text(render_markdown_summary(report), encoding="utf-8")
    log(f"[5/6] Wrote {run_dir}/report.json (+ figures, predictions, error analysis)")
    log(f"[6/6] Saved model -> {model_path} (copy at {latest})")
    return RunResult(run_id, run_dir, model_path, report)
