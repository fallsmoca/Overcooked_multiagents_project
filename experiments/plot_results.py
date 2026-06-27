#!/usr/bin/env python
"""Plot evaluation summaries produced by run_overcooked_experiment.py."""

import argparse
import json
from pathlib import Path


def load_summaries(runs_dir):
    summaries = []
    for path in Path(runs_dir).glob("*/results/eval_summary.json"):
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    return sorted(summaries, key=lambda row: row["experiment"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="runs/eval_bar.png")
    args = parser.parse_args()

    summaries = load_summaries(args.runs_dir)
    if not summaries:
        raise SystemExit(f"No eval_summary.json files found under {args.runs_dir}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib`."
        ) from exc

    labels = [
        f"{row['layout']}\n{row['ego']}-{row['alt']}\n{row['reward_profile']}\nseed {row['seed']}"
        for row in summaries
    ]
    means = [row["mean_sparse_return"] for row in summaries]
    errors = [row["std_sparse_return"] for row in summaries]

    width = max(8, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(labels)), means, yerr=errors, capsize=4, color="#4C78A8")
    ax.set_ylabel("Mean sparse return")
    ax.set_title("Overcooked evaluation results")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
