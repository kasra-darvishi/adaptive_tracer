#!/usr/bin/env python3
"""Plot CE / anomaly-score distributions from per-sequence score dumps.

Expected CSV format:
  split,label,score

Example rows:
  valid_id,normal,0.42
  test_ood_cpu,anomaly,1.73
  test_ood_cpu,anomaly,2.11

Usage:
  python tools/plot_score_distributions.py path/to/scores.csv --outdir logs/score_plots
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "This script requires numpy and matplotlib. Install them with: pip install numpy matplotlib"
    ) from exc


def load_scores(path: Path):
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"split", "label", "score"}
        if not required.issubset(reader.fieldnames or set()):
            raise SystemExit("Input CSV must contain columns: split,label,score")
        for row in reader:
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            grouped[(row["split"], row["label"])].append(score)
    return grouped


def plot_overlay(ax, normal_scores, anomaly_scores, title, bins=100):
    ax.hist(normal_scores, bins=bins, alpha=0.55, density=True, label="Normal")
    ax.hist(anomaly_scores, bins=bins, alpha=0.55, density=True, label="Anomaly")
    ax.set_title(title)
    ax.set_xlabel("Sequence CE / anomaly score")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_multi(groups, outdir: Path, bins=100):
    normal_by_split = {
        split: vals
        for (split, label), vals in groups.items()
        if label.lower() == "normal"
        for split in [split]
    }
    anomaly_by_split = {
        split: vals
        for (split, label), vals in groups.items()
        if label.lower() == "anomaly"
        for split in [split]
    }

    if not normal_by_split:
        raise SystemExit("No normal rows found in score CSV")

    if "valid_id" in normal_by_split:
        global_normal = normal_by_split["valid_id"]
        normal_ref_name = "valid_id"
    elif "test_id" in normal_by_split:
        global_normal = normal_by_split["test_id"]
        normal_ref_name = "test_id"
    else:
        first_split = sorted(normal_by_split)[0]
        global_normal = normal_by_split[first_split]
        normal_ref_name = first_split

    summary_rows = ["split,label,count,mean,median,std,min,max"]
    for (split, label), vals in sorted(groups.items()):
        arr = np.array(vals, dtype=float)
        summary_rows.append(
            f"{split},{label},{len(arr)},{arr.mean():.6f},{np.median(arr):.6f},{arr.std():.6f},{arr.min():.6f},{arr.max():.6f}"
        )

    (outdir / "score_distribution_summary.csv").write_text("\n".join(summary_rows), encoding="utf-8")

    anomaly_items = sorted(anomaly_by_split.items())
    all_anomaly = [score for _, vals in anomaly_items for score in vals]

    n_panels = len(anomaly_items) + (1 if all_anomaly else 0)
    ncols = 3
    nrows = max(1, math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    panel_idx = 0
    for split, anomaly_scores in anomaly_items:
        plot_overlay(axes[panel_idx], global_normal, anomaly_scores, f"{normal_ref_name} vs {split}", bins=bins)
        panel_idx += 1

    if all_anomaly:
        plot_overlay(axes[panel_idx], global_normal, all_anomaly, f"{normal_ref_name} vs all anomalies", bins=bins)
        panel_idx += 1

    for ax in axes[panel_idx:]:
        ax.axis("off")

    fig.suptitle("Per-sequence CE / anomaly-score distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(outdir / "score_distributions_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot CE/anomaly-score distributions from per-sequence score dumps")
    ap.add_argument("score_csv", type=Path, help="CSV with columns split,label,score")
    ap.add_argument("--outdir", type=Path, default=Path("score_plots"), help="Output directory")
    ap.add_argument("--bins", type=int, default=100)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    groups = load_scores(args.score_csv)
    plot_multi(groups, args.outdir, bins=args.bins)
    print(f"Saved combined score distribution plot to: {args.outdir / 'score_distributions_combined.png'}")
    print(f"Saved summary CSV to: {args.outdir / 'score_distribution_summary.csv'}")


if __name__ == "__main__":
    main()
