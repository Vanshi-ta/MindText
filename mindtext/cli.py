"""Command-line interface: python -m mindtext <command>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import DISCLAIMER, __version__
from .config import Config, DataConfig
from .data import load_table, save_quality_report, validate_and_clean
from .errors import DataError, MindTextError
from .evaluation import compute_metrics
from .ingest import build_acoustic_features, prepare_reddit
from .persistence import load_bundle
from .reporting import find_report, format_report_text, read_json, write_json
from .synthetic import make_tabular_dataset, make_text_dataset
from .training import run_experiment


def _data_cfg_from_args(args, base: DataConfig | None = None) -> DataConfig:
    if getattr(args, "config", None):
        base = Config.from_yaml(args.config).data
    cfg = base or DataConfig()
    for attr, key in (("task", "task"), ("text_column", "text_column"), ("label_column", "label_column"), ("id_column", "id_column")):
        v = getattr(args, attr, None)
        if v:
            setattr(cfg, key, v)
    return cfg


def cmd_data_check(args) -> int:
    cfg = _data_cfg_from_args(args)
    df = load_table(args.data, cfg)
    _, rep = validate_and_clean(df, cfg, source_path=args.data)
    print(rep.format_text())
    if args.out:
        save_quality_report(rep, args.out)
        print(f"summary written to {args.out}")
    return 1 if rep.has_errors else 0


def cmd_train(args) -> int:
    cfg = Config.from_yaml(args.config)
    if args.run_name:
        cfg.output.run_name = args.run_name
    res = run_experiment(cfg, data_path=args.data)
    print()
    print(format_report_text(res.report))
    return 0


def cmd_evaluate(args) -> int:
    bundle = load_bundle(args.model)
    cfg = DataConfig(
        task=bundle.task, label_column=args.label_column, id_column=args.id_column,
        text_column=args.text_column or bundle.text_column or "text",
        feature_columns=bundle.feature_columns, allowed_labels=bundle.labels,
        positive_label=bundle.positive_label, dedup_policy="report", invalid_label_policy="error",
    )
    df, rep = validate_and_clean(load_table(args.data, cfg), cfg, source_path=args.data)
    if rep.has_errors:
        raise DataError("Evaluation data failed validation:\n" + rep.format_text())
    X = df["text"].tolist() if bundle.task == "text" else df[bundle.feature_columns]
    out = bundle.predict_frame(X)
    score = out["score"].to_numpy() if bundle.positive_label else None
    m = compute_metrics(df["label"], out["prediction"], bundle.labels, score, bundle.positive_label)
    print(f"Evaluated {m['n']} records from {args.data}")
    print(f"accuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} weighted_f1={m['weighted_f1']:.3f}")
    for lab, v in m["per_class"].items():
        print(f"  {lab:<18} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
    if "binary" in m:
        b = m["binary"]
        print(f"  positive='{b['positive_label']}': TP={b['tp']} FP={b['fp']} TN={b['tn']} FN={b['fn']} ROC-AUC={b['roc_auc']} PR-AUC={b['pr_auc']}")
    print("NOTE: scores are only meaningful if this data was NOT used for training or model selection.")
    if args.out:
        write_json({"model": str(args.model), "data": str(args.data), "metrics": m}, args.out)
        print(f"metrics written to {args.out}")
    return 0


def cmd_predict(args) -> int:
    bundle = load_bundle(args.model)
    if bundle.task == "text":
        if args.text:
            texts, ids = list(args.text), [f"arg-{i}" for i in range(len(args.text))]
        elif args.input:
            d = pd.read_csv(args.input) if Path(args.input).exists() else None
            if d is None:
                raise DataError(f"Input file not found: {args.input}")
            col = args.text_column or bundle.text_column or "text"
            if col not in d.columns:
                raise DataError(f"Column '{col}' not in {args.input}. Columns: {list(d.columns)}")
            texts, ids = d[col].fillna("").astype(str).tolist(), [f"row-{i}" for i in range(len(d))]
        else:
            raise DataError("Provide --text 'some text' (repeatable) or --input file.csv.")
        res = bundle.predict_frame(texts)
    else:
        if not args.input:
            raise DataError("Tabular models need --input file.csv containing the training feature columns.")
        p = Path(args.input)
        if not p.exists():
            raise DataError(f"Input file not found: {p}")
        d = pd.read_csv(p)
        res = bundle.predict_frame(d)
        ids = [f"row-{i}" for i in range(len(d))]
    res.insert(0, "record", ids)
    if args.json:
        print(json.dumps(res.to_dict(orient="records"), indent=2, default=float))
    else:
        for _, r in res.iterrows():
            print(f"{r['record']}: predicted class = {r['prediction']}  (score={r['score']:.3f}, {r['score_kind']})")
    if args.out:
        res.to_csv(args.out, index=False)
    print(f"\n{DISCLAIMER}", file=sys.stderr if args.json else sys.stdout)
    return 0


def cmd_report(args) -> int:
    path = Path(args.path) if args.path else find_report(args.reports_dir, None if args.latest else args.run_id)
    rep = read_json(path)
    print(json.dumps(rep, indent=2) if args.json else format_report_text(rep))
    return 0


def cmd_make_sample(args) -> int:
    df = make_text_dataset(args.n, seed=args.seed) if args.task == "text" else make_tabular_dataset(seed=args.seed)
    if args.task == "text":
        df = df[["id", "text", "label"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote SYNTHETIC {args.task} sample ({len(df)} rows) to {out}. Scores on it are not meaningful.")
    return 0


def cmd_prepare_reddit(args) -> int:
    s = prepare_reddit(args.raw_dir, args.out, text_column=args.text_column, strict=args.strict, hash_salt=args.salt)
    print(f"Wrote {s['rows_written']} rows to {args.out}; per label: {s['per_label']}")
    if s["unmatched_files"]:
        print(f"WARNING: {len(s['unmatched_files'])} file(s) matched no label rule and were skipped: {s['unmatched_files'][:10]}")
    if s["suspected_copies"]:
        print(f"WARNING: file(s) named like copies were included: {s['suspected_copies']} (duplicates are handled by data.dedup_policy)")
    write_json(s, Path(args.out).with_suffix(".ingest.json"))
    return 0


def cmd_build_features(args) -> int:
    s = build_acoustic_features(args.raw_dir, args.labels_dir, args.out, tuple(args.sources),
                                sentinels=tuple(args.sentinel), nan_fill=args.nan_fill)
    print(f"Wrote {s['rows']} participants x {s['n_features']} features to {args.out}")
    if s["missing_participants"]:
        print(f"WARNING: raw files missing for participants {s['missing_participants']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mindtext", description="Reproducible mental-health text / acoustic-feature classification (research use only).")
    p.add_argument("--version", action="version", version=f"mindtext {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("data-check", help="validate a dataset and report quality issues")
    s.add_argument("--data", required=True)
    s.add_argument("--config")
    s.add_argument("--task", choices=["text", "tabular"])
    s.add_argument("--text-column"); s.add_argument("--label-column"); s.add_argument("--id-column")
    s.add_argument("--out", help="write the quality summary JSON here")
    s.set_defaults(func=cmd_data_check)

    s = sub.add_parser("train", help="train/evaluate the configured models")
    s.add_argument("--config", required=True)
    s.add_argument("--data", help="override data.path from the config")
    s.add_argument("--run-name")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="evaluate a saved model on a labelled dataset")
    s.add_argument("--model", required=True); s.add_argument("--data", required=True)
    s.add_argument("--label-column", default="label"); s.add_argument("--text-column"); s.add_argument("--id-column")
    s.add_argument("--out")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("predict", help="illustrative predictions (not a diagnosis)")
    s.add_argument("--model", required=True)
    s.add_argument("--text", action="append")
    s.add_argument("--input"); s.add_argument("--text-column"); s.add_argument("--out"); s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("report", help="show an experiment report")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true"); g.add_argument("--run-id"); g.add_argument("--path")
    s.add_argument("--reports-dir", default="reports"); s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("make-sample", help="write a small SYNTHETIC demo dataset")
    s.add_argument("--task", choices=["text", "tabular"], default="text")
    s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=40, help="rows per class (text)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_make_sample)

    s = sub.add_parser("prepare-reddit", help="combine raw Reddit CSV files into one labelled CSV")
    s.add_argument("--raw-dir", required=True); s.add_argument("--out", required=True)
    s.add_argument("--text-column", default="selftext"); s.add_argument("--strict", action="store_true")
    s.add_argument("--salt", default="", help="salt for hashing author names, if present")
    s.set_defaults(func=cmd_prepare_reddit)

    s = sub.add_parser("build-acoustic-features", help="aggregate frame-level openSMILE files per participant")
    s.add_argument("--raw-dir", required=True); s.add_argument("--labels-dir", required=True); s.add_argument("--out", required=True)
    s.add_argument("--sources", nargs="+", default=["egemaps"], choices=["egemaps", "mfcc"])
    s.add_argument("--sentinel", nargs="*", type=float, default=[-201.0], help="values treated as missing")
    s.add_argument("--nan-fill", type=float, default=None)
    s.set_defaults(func=cmd_build_features)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (MindTextError, FileNotFoundError, PermissionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
