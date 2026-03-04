import argparse
import ast
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "eval" / "eval_rag.py"
HISTORY_CSV = ROOT / "eval" / "metrics_history.csv"
REPORTS_DIR = ROOT / "eval" / "reports"


DEFAULT_CONFIGS = {
    "small_k4": {"gold": "eval/gold_qa_small.jsonl", "k": 4, "no_llm": True},
    "expanded_k4": {"gold": "eval/gold_qa.jsonl", "k": 4, "no_llm": True},
    "expanded_k8": {"gold": "eval/gold_qa.jsonl", "k": 8, "no_llm": True},
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _ensure_storage() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_CSV.exists():
        with HISTORY_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "label",
                    "notes",
                    "git_sha",
                    "config",
                    "gold",
                    "k",
                    "no_llm",
                    "N",
                    "recall",
                    "mrr",
                    "faithfulness",
                    "citation_rate",
                    "instrument_hit_rate",
                    "llm_errors",
                ],
            )
            writer.writeheader()


def _parse_eval_metrics(stdout: str) -> dict:
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    candidates = [x for x in lines if x.startswith("{") and "'N':" in x]
    if not candidates:
        raise RuntimeError("Could not parse eval output. No metrics dictionary found.")
    raw = ast.literal_eval(candidates[-1])

    recall_key = next((k for k in raw if str(k).startswith("Recall@")), None)
    mrr_key = next((k for k in raw if str(k).startswith("MRR@")), None)

    return {
        "N": raw.get("N", 0),
        "recall": float(raw.get(recall_key, 0.0)) if recall_key else 0.0,
        "mrr": float(raw.get(mrr_key, 0.0)) if mrr_key else 0.0,
        "faithfulness": float(raw.get("Faithfulness(proxy)", 0.0)),
        "citation_rate": float(raw.get("Answer with citation rate", 0.0)),
        "instrument_hit_rate": float(raw.get("Instrument hit rate", 0.0)),
        "llm_errors": int(raw.get("LLM errors", 0)),
    }


def _run_eval(python_exe: str, gold: str, k: int, no_llm: bool) -> dict:
    cmd = [python_exe, str(EVAL_SCRIPT), "--gold", gold, "--k", str(k)]
    if no_llm:
        cmd.append("--no-llm")

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Eval failed for gold={gold}, k={k}, no_llm={no_llm}.\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )
    return _parse_eval_metrics(proc.stdout)


def _append_rows(rows: list[dict]) -> None:
    _ensure_storage()
    with HISTORY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pd.read_csv(HISTORY_CSV).columns.tolist())
        for row in rows:
            writer.writerow(row)


def _load_history() -> pd.DataFrame:
    _ensure_storage()
    df = pd.read_csv(HISTORY_CSV)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "config"]).reset_index(drop=True)
    return df


def _plot_history(df: pd.DataFrame) -> None:
    if df.empty:
        return

    metrics = [
        ("recall", "Recall"),
        ("mrr", "MRR"),
        ("faithfulness", "Faithfulness"),
        ("instrument_hit_rate", "Instrument hit rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        for config_name, g in df.groupby("config"):
            g = g.sort_values("timestamp")
            ax.plot(g["timestamp"], g[col], marker="o", label=config_name)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    out_path = REPORTS_DIR / "metrics_history.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_tables(df: pd.DataFrame) -> None:
    if df.empty:
        return

    def _to_md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_empty_"
        cols = list(frame.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in frame.iterrows():
            vals = [str(row[c]) for c in cols]
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep, *rows])

    latest = df.sort_values("timestamp").groupby("config", as_index=False).tail(1)
    latest = latest[[
        "timestamp", "label", "config", "N", "recall", "mrr", "faithfulness", "citation_rate", "instrument_hit_rate", "llm_errors"
    ]].sort_values("config")
    latest.to_csv(REPORTS_DIR / "latest_metrics.csv", index=False)

    first = df.sort_values("timestamp").groupby("config", as_index=False).head(1)
    last = df.sort_values("timestamp").groupby("config", as_index=False).tail(1)
    merged = first.merge(last, on="config", suffixes=("_before", "_after"))

    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "config": r["config"],
            "N": int(r["N_after"]),
            "recall_before": float(r["recall_before"]),
            "recall_after": float(r["recall_after"]),
            "mrr_before": float(r["mrr_before"]),
            "mrr_after": float(r["mrr_after"]),
            "faithfulness_before": float(r["faithfulness_before"]),
            "faithfulness_after": float(r["faithfulness_after"]),
            "instrument_before": float(r["instrument_hit_rate_before"]),
            "instrument_after": float(r["instrument_hit_rate_after"]),
        })

    ba = pd.DataFrame(rows).sort_values("config")
    ba.to_csv(REPORTS_DIR / "before_after.csv", index=False)

    md_lines = [
        "# Metrics Report",
        "",
        "## Latest metrics",
        "",
        _to_md_table(latest),
        "",
        "## Before vs After",
        "",
        _to_md_table(ba),
        "",
        "## Plot",
        "",
        "![Metrics history](metrics_history.png)",
    ]
    (REPORTS_DIR / "metrics_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def render_reports(args: argparse.Namespace) -> None:
    df = _load_history()
    _plot_history(df)
    _write_tables(df)
    print(f"Reports rebuilt from {HISTORY_CSV}")
    print(f"Reports: {REPORTS_DIR}")


def run_capture(args: argparse.Namespace) -> None:
    rows = []
    ts = _now_iso()
    git_sha = _git_sha()

    config_names = args.configs or list(DEFAULT_CONFIGS.keys())
    for cfg in config_names:
        if cfg not in DEFAULT_CONFIGS:
            raise ValueError(f"Unknown config: {cfg}. Available: {', '.join(DEFAULT_CONFIGS.keys())}")
        c = DEFAULT_CONFIGS[cfg]
        m = _run_eval(args.python, c["gold"], c["k"], c["no_llm"])
        rows.append({
            "timestamp": ts,
            "label": args.label,
            "notes": args.notes,
            "git_sha": git_sha,
            "config": cfg,
            "gold": c["gold"],
            "k": c["k"],
            "no_llm": c["no_llm"],
            **m,
        })

    _append_rows(rows)
    df = _load_history()
    _plot_history(df)
    _write_tables(df)
    print(f"Saved {len(rows)} rows to {HISTORY_CSV}")
    print(f"Reports: {REPORTS_DIR}")


def add_manual(args: argparse.Namespace) -> None:
    row = {
        "timestamp": args.timestamp or _now_iso(),
        "label": args.label,
        "notes": args.notes,
        "git_sha": args.git_sha,
        "config": args.config,
        "gold": args.gold,
        "k": args.k,
        "no_llm": args.no_llm,
        "N": args.n,
        "recall": args.recall,
        "mrr": args.mrr,
        "faithfulness": args.faithfulness,
        "citation_rate": args.citation_rate,
        "instrument_hit_rate": args.instrument_hit_rate,
        "llm_errors": args.llm_errors,
    }
    _append_rows([row])
    df = _load_history()
    _plot_history(df)
    _write_tables(df)
    print(f"Manual row added to {HISTORY_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track RAG metrics over time and build plots.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run eval configs, save metrics and build plots")
    run_p.add_argument("--label", required=True, help="Run label, e.g. structured-lookup-fix")
    run_p.add_argument("--notes", default="", help="What changed in this run")
    run_p.add_argument("--python", default=sys.executable, help="Python executable path")
    run_p.add_argument("--configs", nargs="*", default=None, help="Subset of configs to run")
    run_p.set_defaults(func=run_capture)

    manual_p = sub.add_parser("add-manual", help="Add manual baseline row")
    manual_p.add_argument("--timestamp", default="", help="ISO timestamp, optional")
    manual_p.add_argument("--label", required=True)
    manual_p.add_argument("--notes", default="")
    manual_p.add_argument("--git-sha", default="")
    manual_p.add_argument("--config", required=True)
    manual_p.add_argument("--gold", required=True)
    manual_p.add_argument("--k", type=int, required=True)
    manual_p.add_argument("--no-llm", action="store_true")
    manual_p.add_argument("--n", type=int, required=True)
    manual_p.add_argument("--recall", type=float, required=True)
    manual_p.add_argument("--mrr", type=float, required=True)
    manual_p.add_argument("--faithfulness", type=float, required=True)
    manual_p.add_argument("--citation-rate", type=float, required=True)
    manual_p.add_argument("--instrument-hit-rate", type=float, required=True)
    manual_p.add_argument("--llm-errors", type=int, default=0)
    manual_p.set_defaults(func=add_manual)

    render_p = sub.add_parser("render", help="Rebuild plots/tables from existing history")
    render_p.set_defaults(func=render_reports)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
