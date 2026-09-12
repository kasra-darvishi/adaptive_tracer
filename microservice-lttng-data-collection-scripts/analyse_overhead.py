#!/usr/bin/env python3
"""
analyse_overhead.py — Compute LMAT overhead table from load_results.csv files
==============================================================================

Reads all load_results.csv files from each experimental condition, computes
P50 / P95 / P99 response time and throughput, and prints a markdown table
ready to paste into the paper.

Usage:
    python3 analyse_overhead.py \
        --baseline_dir   ~/experiments/baseline \
        --lttng_only_dir ~/experiments/lttng_only \
        --sync_dir       ~/experiments/lmat_sync \
        --async_dir      ~/experiments/lmat_async \
        --output         overhead_table.md

Each <condition_dir> is scanned for run*/load_results.csv files;
all runs for that condition are pooled before computing percentiles.

Output CSV is also written to <output>.csv for use in LaTeX/Excel.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


###############################################################################
# CLI
###############################################################################

def get_args():
    p = argparse.ArgumentParser(
        description="Compute LMAT overhead statistics from load_results.csv files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline_dir",   default=None,
                   help="experiments/baseline — pure baseline (no tracing)")
    p.add_argument("--lttng_only_dir", default=None,
                   help="experiments/lttng_only — LTTng on, no LMAT")
    p.add_argument("--sync_dir",       default=None,
                   help="experiments/lmat_sync — LTTng + LMAT synchronous")
    p.add_argument("--async_dir",      default=None,
                   help="experiments/lmat_async — LTTng + LMAT asynchronous")
    p.add_argument("--output",         default="overhead_table.md",
                   help="Output markdown file path")
    p.add_argument("--exclude_setup",  action="store_true",
                   help="Exclude register/login/setup_* scenarios from latency stats "
                        "(these run once per user and skew p99)")
    p.add_argument("--min_latency_ms", type=float, default=0.0,
                   help="Discard requests with latency below this threshold (ms)")
    p.add_argument("--max_latency_ms", type=float, default=60_000.0,
                   help="Discard requests with latency above this threshold (ms) — removes timeouts")
    return p.parse_args()


###############################################################################
# Data loading
###############################################################################

SETUP_SCENARIOS = {"register", "setup_address", "setup_card", "login"}


def find_csv_files(root_dir: str):
    """Return all load_results.csv files found under root_dir."""
    root = Path(root_dir).expanduser()
    if not root.exists():
        print(f"  [WARN] Directory not found: {root}", file=sys.stderr)
        return []
    files = sorted(root.rglob("load_results.csv"))
    if not files:
        print(f"  [WARN] No load_results.csv found under {root}", file=sys.stderr)
    return files


def load_latencies(csv_files, exclude_setup: bool,
                   min_ms: float, max_ms: float):
    """
    Read all CSV files and return:
        latencies_ms : np.ndarray of all successful request latencies
        n_total      : int, total requests (including failed)
        n_success    : int, successful requests
        duration_s   : float, estimated total experiment duration (seconds)
    """
    latencies  = []
    n_total    = 0
    n_success  = 0
    t_min      = float("inf")
    t_max      = float("-inf")

    for fpath in csv_files:
        try:
            with open(fpath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    n_total += 1
                    # Skip setup-phase requests if requested
                    if exclude_setup and row.get("scenario", "") in SETUP_SCENARIOS:
                        continue
                    # Skip failed requests
                    success_str = row.get("success", "True").strip().lower()
                    if success_str not in ("true", "1", "yes"):
                        continue
                    try:
                        lat = float(row["latency_ms"])
                    except (KeyError, ValueError):
                        continue
                    if lat < min_ms or lat > max_ms:
                        continue
                    latencies.append(lat)
                    n_success += 1
                    # Track timestamp range for duration estimate
                    ts_str = row.get("timestamp", "")
                    if ts_str:
                        try:
                            from datetime import datetime, timezone
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            epoch = ts.timestamp()
                            t_min = min(t_min, epoch)
                            t_max = max(t_max, epoch)
                        except Exception:
                            pass
        except Exception as e:
            print(f"  [WARN] Could not read {fpath}: {e}", file=sys.stderr)

    duration_s = (t_max - t_min) if t_max > t_min else 1.0
    return np.array(latencies, dtype=np.float64), n_total, n_success, duration_s


###############################################################################
# Statistics
###############################################################################

def compute_stats(latencies: np.ndarray, n_total: int, n_success: int,
                  duration_s: float):
    """Return a dict of statistics."""
    if latencies.size == 0:
        return {k: float("nan") for k in
                ("mean", "p50", "p95", "p99", "throughput", "error_rate", "n_requests")}
    return {
        "n_requests": n_success,
        "mean":        float(np.mean(latencies)),
        "p50":         float(np.percentile(latencies, 50)),
        "p95":         float(np.percentile(latencies, 95)),
        "p99":         float(np.percentile(latencies, 99)),
        "throughput":  float(n_success / duration_s),
        "error_rate":  float((n_total - n_success) / max(n_total, 1) * 100),
    }


def overhead_pct(value, baseline):
    """Percentage overhead relative to baseline. Positive = slower/worse."""
    if baseline == 0 or np.isnan(baseline) or np.isnan(value):
        return float("nan")
    return (value - baseline) / baseline * 100.0


###############################################################################
# Formatting helpers
###############################################################################

def fmt(v, decimals=1):
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def fmt_pct(v):
    if np.isnan(v):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


###############################################################################
# Main
###############################################################################

def main():
    args = get_args()

    conditions = [
        ("Baseline (no tracing)",  args.baseline_dir),
        ("LTTng only",             args.lttng_only_dir),
        ("Sync LMAT co-located",   args.sync_dir),
        ("Async LMAT co-located",  args.async_dir),
    ]

    results = {}
    print("\n── Loading experiment data ──────────────────────────────────────")
    for label, root in conditions:
        if root is None:
            print(f"  {label:<30} SKIPPED (--dir not provided)")
            results[label] = None
            continue
        files = find_csv_files(root)
        if not files:
            results[label] = None
            continue
        print(f"  {label:<30} {len(files)} run(s) found")
        lats, n_total, n_success, duration_s = load_latencies(
            files, args.exclude_setup, args.min_latency_ms, args.max_latency_ms)
        stats = compute_stats(lats, n_total, n_success, duration_s)
        results[label] = stats
        print(f"    → n={n_success:,}  p50={fmt(stats['p50'])} ms  "
              f"p95={fmt(stats['p95'])} ms  "
              f"throughput={fmt(stats['throughput'])} req/s  "
              f"error={fmt(stats['error_rate'])}%")

    # ── Compute overhead relative to baseline ─────────────────────────────────
    baseline_stats = results.get("Baseline (no tracing)")
    b_p50  = baseline_stats["p50"]  if baseline_stats else float("nan")
    b_p95  = baseline_stats["p95"]  if baseline_stats else float("nan")
    b_p99  = baseline_stats["p99"]  if baseline_stats else float("nan")
    b_tput = baseline_stats["throughput"] if baseline_stats else float("nan")

    # ── Build markdown table ──────────────────────────────────────────────────
    header = (
        "| Condition              | Requests | Throughput (req/s) | "
        "P50 (ms) | P95 (ms) | P99 (ms) | P95 overhead | Throughput overhead |"
    )
    sep = (
        "|---|---|---|---|---|---|---|---|"
    )

    rows = [header, sep]
    for label, _ in conditions:
        s = results.get(label)
        if s is None:
            rows.append(f"| {label} | — | — | — | — | — | — | — |")
            continue
        p95_oh  = overhead_pct(s["p95"],  b_p95)
        tput_oh = overhead_pct(s["throughput"], b_tput)
        # Throughput: lower is worse, so flip sign for "overhead" display
        tput_oh_str = fmt_pct(-tput_oh) if not np.isnan(tput_oh) else "—"
        rows.append(
            f"| {label} "
            f"| {s['n_requests']:,} "
            f"| {fmt(s['throughput'])} "
            f"| {fmt(s['p50'])} "
            f"| {fmt(s['p95'])} "
            f"| {fmt(s['p99'])} "
            f"| {fmt_pct(p95_oh)} "
            f"| {tput_oh_str} |"
        )

    table_md = "\n".join(rows)

    # ── Build full markdown output ────────────────────────────────────────────
    exclude_note = (
        "\n> Setup-phase requests (register, login, setup_address, setup_card) "
        "excluded from latency statistics.\n"
        if args.exclude_setup else ""
    )

    output_md = f"""# LMAT Overhead Measurement Results

SockShop (Weaveworks microservices benchmark) — 200 concurrent virtual users — 300s runs.  
Anomaly score: 0.7 × event cross-entropy + 0.3 × latency cross-entropy.  
LMAT model runs on CPU (GCP VM, 12 vCPU, no GPU) using 2 threads to minimise interference.
{exclude_note}
## Response Time and Throughput

{table_md}

> All latencies are end-to-end HTTP response time measured at the load generator.  
> P95 overhead and throughput overhead are relative to the **Baseline (no tracing)** condition.  
> Positive P95 overhead = slower; negative throughput overhead = fewer requests served.
"""

    out_path = Path(args.output).expanduser()
    out_path.write_text(output_md, encoding="utf-8")
    print(f"\n── Markdown table written to: {out_path}")

    # ── Also write CSV ─────────────────────────────────────────────────────────
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n_requests", "throughput_req_s",
                         "p50_ms", "p95_ms", "p99_ms",
                         "p95_overhead_pct", "throughput_overhead_pct"])
        for label, _ in conditions:
            s = results.get(label)
            if s is None:
                writer.writerow([label] + ["nan"] * 7)
                continue
            writer.writerow([
                label, s["n_requests"],
                f"{s['throughput']:.2f}",
                f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{s['p99']:.1f}",
                f"{overhead_pct(s['p95'], b_p95):.2f}",
                f"{overhead_pct(s['throughput'], b_tput):.2f}",
            ])
    print(f"── CSV written to:      {csv_path}")

    # ── Print table to stdout ──────────────────────────────────────────────────
    print("\n" + output_md)


if __name__ == "__main__":
    main()
