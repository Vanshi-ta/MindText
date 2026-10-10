"""Run reports: JSON serialisation, figures generated from real results, summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from . import DISCLAIMER  # noqa: E402
from .errors import MindTextError  # noqa: E402


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_default, allow_nan=False), encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise MindTextError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Locating runs
# --------------------------------------------------------------------------
def list_runs(reports_dir: str | Path) -> list[Path]:
    d = Path(reports_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and (p / "report.json").exists())


def find_report(reports_dir: str | Path, run_id: str | None = None) -> Path:
    runs = list_runs(reports_dir)
    if not runs:
        raise MindTextError(f"No experiment reports found in '{reports_dir}'. Run `python -m mindtext train` first.")
    if run_id is None:
        return runs[-1] / "report.json"  # run ids start with a UTC timestamp -> lexicographic order
    for r in runs:
        if r.name == run_id:
            return r / "report.json"
    raise MindTextError(f"Run '{run_id}' not found in '{reports_dir}'. Available: {[r.name for r in runs]}")


# --------------------------------------------------------------------------
# Figures (always drawn from computed results, never from literals)
# --------------------------------------------------------------------------
def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_class_distribution(class_counts: dict[str, dict[str, int]], path: Path) -> str:
    splits = [s for s, c in class_counts.items() if c]
    labels = sorted({l for s in splits for l in class_counts[s]})
    x = np.arange(len(labels))
    w = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 3), 4))
    for i, s in enumerate(splits):
        ax.bar(x + i * w, [class_counts[s].get(l, 0) for l in labels], w, label=s)
    ax.set_xticks(x + w * (len(splits) - 1) / 2, labels, rotation=20, ha="right")
    ax.set_ylabel("records")
    ax.set_title("Class distribution per split")
    ax.legend()
    return _save(fig, path)


def plot_model_comparison(models: list[dict], split: str, path: Path) -> str:
    names = [m["name"] for m in models]
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names) + 2), 4.2))
    for i, k in enumerate(keys):
        ax.bar(x + i * w, [m[split]["metrics"][k] for m in models], w, label=k)
    ax.set_xticks(x + w, names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Model comparison ({split} split)")
    ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], path: Path, title: str, normalize: bool = True) -> str:
    a = np.asarray(cm, dtype=float)
    shown = a / a.sum(axis=1, keepdims=True).clip(min=1) if normalize else a
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(labels) + 2), max(4, 0.8 * len(labels) + 1.5)))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{int(a[i, j])}" + (f"\n{shown[i, j]:.0%}" if normalize else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if shown[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_per_class(per_class: dict[str, dict], path: Path, title: str) -> str:
    labels = list(per_class)
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels) + 2), 4))
    for i, k in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + i * w, [per_class[l][k] for l in labels], w, label=k)
    ax.set_xticks(x + w, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_precision_recall(y_true_bin: np.ndarray, scores: np.ndarray, path: Path, title: str) -> str:
    prec, rec, _ = precision_recall_curve(y_true_bin, scores)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot(rec, prec)
    ax.axhline(float(np.mean(y_true_bin)), ls="--", c="grey", label="prevalence (no-skill)")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def make_figures(report: dict, preds: dict[str, pd.DataFrame], run_dir: Path) -> list[str]:
    """``preds`` maps '<model>|<split>' to prediction frames saved by training."""
    fig_dir = run_dir / "figures"
    out = [plot_class_distribution(report["split"]["class_counts"], fig_dir / "class_distribution.png")]
    models = report["models"]
    for split in ("val", "test"):
        if all(m.get(split) for m in models):
            out.append(plot_model_comparison(models, split, fig_dir / f"model_comparison_{split}.png"))
    sel = next(m for m in models if m["name"] == report["selected_model"])
    t = sel["test"]["metrics"]
    out.append(plot_confusion_matrix(t["confusion_matrix"]["matrix"], t["confusion_matrix"]["labels"], fig_dir / "confusion_matrix_test.png", f"{sel['name']} - test"))
    out.append(plot_per_class(t["per_class"], fig_dir / "per_class_test.png", f"{sel['name']} - per-class (test)"))
    pos = report["data"].get("positive_label")
    frame = preds.get(f"{sel['name']}|test")
    if pos is not None and frame is not None and frame["score"].notna().all():
        yb = (frame["y_true"] == pos).astype(int).to_numpy()
        if 0 < yb.sum() < len(yb):
            out.append(plot_precision_recall(yb, frame["score"].to_numpy(), fig_dir / "precision_recall_test.png", f"{sel['name']} - PR (test)"))
    return out


# --------------------------------------------------------------------------
# Text summaries
# --------------------------------------------------------------------------
def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def format_report_text(report: dict) -> str:
    d, sp = report["data"], report["split"]
    lines = [
        f"Run {report['run_id']}   ({report['created_at']})",
        f"Task: {report['task']}   data: {d['path']}" + (f"   [{d['description']}]" if d.get("description") else ""),
        f"Split: {sp['strategy']} seed={sp['seed']} sizes={sp['sizes']}",
        f"Selection rule: {report['selection_rule']}",
        "",
        f"{'model':<22}{'prov.':<9}{'val macroF1':>12}{'test acc':>10}{'test mF1':>10}{'test wF1':>10}",
    ]
    for m in report["models"]:
        star = " *" if m["name"] == report["selected_model"] else ""
        v = m.get("val")
        lines.append(
            f"{m['name'] + star:<22}{m['provenance']:<9}"
            f"{_fmt(v['metrics']['macro_f1']) if v else 'n/a':>12}"
            f"{_fmt(m['test']['metrics']['accuracy']):>10}{_fmt(m['test']['metrics']['macro_f1']):>10}{_fmt(m['test']['metrics']['weighted_f1']):>10}"
        )
    lines.append("(* = selected on the validation split; test scores are reported once, after selection)")
    base = report.get("baselines", {})
    if "all_positive" in base:
        ap = base["all_positive"]
        lines.append(f"Context (test): all-positive baseline F1={_fmt(ap['f1'])}, precision={_fmt(ap['precision'])}; majority accuracy={_fmt(base['majority']['accuracy'])}")
    sel = next(m for m in report["models"] if m["name"] == report["selected_model"])
    ci = sel.get("test_ci") or {}
    if ci:
        lines.append(f"Selected model test 95% bootstrap CI: macro-F1 {ci['macro_f1'][0]:.3f}-{ci['macro_f1'][1]:.3f}, accuracy {ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f}")
    for w in report.get("warnings", []):
        lines.append(f"WARNING: {w}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def render_markdown_summary(report: dict) -> str:
    sel = next(m for m in report["models"] if m["name"] == report["selected_model"])
    tm = sel["test"]["metrics"]
    rows = "\n".join(
        f"| {m['name']}{' **(selected)**' if m['name'] == report['selected_model'] else ''} | {m['provenance']} | "
        f"{_fmt(m['val']['metrics']['macro_f1']) if m.get('val') else 'n/a'} | {_fmt(m['test']['metrics']['accuracy'])} | "
        f"{_fmt(m['test']['metrics']['macro_f1'])} | {_fmt(m['test']['metrics']['weighted_f1'])} |"
        for m in report["models"]
    )
    per = "\n".join(
        f"| {l} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} | {v['support']} |" for l, v in tm["per_class"].items()
    )
    warn = "\n".join(f"- {w}" for w in report.get("warnings", [])) or "- none"
    return f"""# Run {report['run_id']}

> {DISCLAIMER}

- Task: `{report['task']}`; data: `{report['data']['path']}` {('- ' + report['data']['description']) if report['data'].get('description') else ''}
- Split: `{report['split']['strategy']}` (seed {report['split']['seed']}), sizes {report['split']['sizes']}
- Selection rule: {report['selection_rule']}

| model | provenance | val macro-F1 | test accuracy | test macro-F1 | test weighted-F1 |
|---|---|---|---|---|---|
{rows}

## Selected model: `{sel['name']}` - per-class test metrics

| class | precision | recall | F1 | support |
|---|---|---|---|---|
{per}

## Warnings
{warn}
"""
