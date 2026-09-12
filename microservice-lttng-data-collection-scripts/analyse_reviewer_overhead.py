#!/usr/bin/env python3
"""
Summarise reviewer-facing overhead results from a load sweep.

This script directly answers:
- What are P95/P99 latencies without LMAT vs with LMAT enabled?
- What is the maximum throughput without LMAT vs with LMAT enabled?

Expected directory layout (produced by run_reviewer_overhead_matrix.sh):
  <experiment_root>/
    baseline/u200_r01/load_results.csv
    lttng_only/u200_r01/load_results.csv
    lmat_async/u200_r01/load_results_with_inference.csv
    ...
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


SETUP_SCENARIOS = {"register", "setup_address", "setup_card", "login"}
RUN_ID_RE = re.compile(r"u(?P<users>\d+)_r(?P<repeat>\d+)")


@dataclass
class RunStats:
    condition: str
    users: int
    repeat: int
    run_id: str
    file_path: Path
    n_total: int
    n_success: int
    duration_s: float
    throughput: float
    error_rate_pct: float
    p50: float
    p95: float
    p99: float
    mean: float


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse VM latency/throughput sweep for reviewer-facing overhead claims",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment_root", required=True, help="Root directory containing baseline/lttng_only/lmat_async runs")
    p.add_argument("--reference_users", type=int, default=200, help="User level to use for the direct latency comparison table")
    p.add_argument("--max_error_rate_pct", type=float, default=1.0, help="Max error rate allowed when selecting maximum throughput")
    p.add_argument("--exclude_setup", action="store_true", help="Exclude register/login/setup_* scenarios from latency stats")
    p.add_argument("--output_prefix", default=None, help="Output file prefix; defaults to <experiment_root>/reviewer_overhead")
    return p.parse_args()


def condition_result_filename(condition: str) -> str:
    if condition in {"baseline", "lttng_only"}:
        return "load_results.csv"
    if condition in {"lmat_async", "lmat_sync"}:
        return "load_results_with_inference.csv"
    raise ValueError(f"Unsupported condition: {condition}")


def iter_result_files(root: Path):
    for condition_dir in root.iterdir():
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        target_name = condition_result_filename(condition) if condition in {"baseline", "lttng_only", "lmat_async", "lmat_sync"} else None
        if target_name is None:
            continue
        for run_dir in sorted(condition_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = RUN_ID_RE.fullmatch(run_dir.name)
            if not m:
                continue
            result_path = run_dir / target_name
            if result_path.exists():
                yield condition, int(m.group("users")), int(m.group("repeat")), run_dir.name, result_path


def parse_timestamp(ts: str) -> float | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def load_run_stats(condition: str, users: int, repeat: int, run_id: str, file_path: Path, exclude_setup: bool) -> RunStats | None:
    latencies = []
    n_total = 0
    n_success = 0
    t_min = math.inf
    t_max = -math.inf

    with file_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if exclude_setup and row.get("scenario", "") in SETUP_SCENARIOS:
                continue

            ts = parse_timestamp(row.get("timestamp", ""))
            if ts is not None:
                t_min = min(t_min, ts)
                t_max = max(t_max, ts)

            success = row.get("success", "").strip().lower() in {"true", "1", "yes"}
            if not success:
                continue
            try:
                lat = float(row["latency_ms"])
            except Exception:
                continue
            latencies.append(lat)
            n_success += 1

    if not latencies:
        return None

    duration_s = (t_max - t_min) if t_max > t_min else 1.0
    lat_arr = np.asarray(latencies, dtype=np.float64)
    throughput = n_success / duration_s
    error_rate_pct = (n_total - n_success) / max(n_total, 1) * 100.0

    return RunStats(
        condition=condition,
        users=users,
        repeat=repeat,
        run_id=run_id,
        file_path=file_path,
        n_total=n_total,
        n_success=n_success,
        duration_s=duration_s,
        throughput=throughput,
        error_rate_pct=error_rate_pct,
        p50=float(np.percentile(lat_arr, 50)),
        p95=float(np.percentile(lat_arr, 95)),
        p99=float(np.percentile(lat_arr, 99)),
        mean=float(np.mean(lat_arr)),
    )


def group_by_condition_users(runs: Iterable[RunStats]):
    grouped: dict[tuple[str, int], list[RunStats]] = defaultdict(list)
    for run in runs:
        grouped[(run.condition, run.users)].append(run)
    return grouped


def mean_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def summarise_group(condition: str, users: int, runs: list[RunStats]):
    return {
        "condition": condition,
        "users": users,
        "repeats": len(runs),
        "throughput_mean": mean_std([r.throughput for r in runs])[0],
        "throughput_std": mean_std([r.throughput for r in runs])[1],
        "error_rate_mean": mean_std([r.error_rate_pct for r in runs])[0],
        "error_rate_std": mean_std([r.error_rate_pct for r in runs])[1],
        "p50_mean": mean_std([r.p50 for r in runs])[0],
        "p50_std": mean_std([r.p50 for r in runs])[1],
        "p95_mean": mean_std([r.p95 for r in runs])[0],
        "p95_std": mean_std([r.p95 for r in runs])[1],
        "p99_mean": mean_std([r.p99 for r in runs])[0],
        "p99_std": mean_std([r.p99 for r in runs])[1],
        "n_success_mean": mean_std([r.n_success for r in runs])[0],
    }


def pick_max_throughput(summary_rows: list[dict], max_error_rate_pct: float):
    best: dict[str, dict] = {}
    for row in summary_rows:
        if row["error_rate_mean"] > max_error_rate_pct:
            continue
        cond = row["condition"]
        if cond not in best or row["throughput_mean"] > best[cond]["throughput_mean"]:
            best[cond] = row
    return best


def fmt(x: float, decimals: int = 1):
    if math.isnan(x):
        return "—"
    return f"{x:.{decimals}f}"


def main():
    args = parse_args()
    root = Path(args.experiment_root).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser() if args.output_prefix else root / "reviewer_overhead"

    runs = []
    for item in iter_result_files(root):
        stat = load_run_stats(*item, exclude_setup=args.exclude_setup)
        if stat is not None:
            runs.append(stat)

    if not runs:
        raise SystemExit(f"No valid run results found under {root}")

    grouped = group_by_condition_users(runs)
    summary_rows = [summarise_group(cond, users, grouped[(cond, users)]) for cond, users in sorted(grouped)]

    detail_csv = output_prefix.with_name(output_prefix.name + "_by_users.csv")
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "users", "repeats",
            "throughput_mean_req_s", "throughput_std_req_s",
            "error_rate_mean_pct", "error_rate_std_pct",
            "p50_mean_ms", "p50_std_ms",
            "p95_mean_ms", "p95_std_ms",
            "p99_mean_ms", "p99_std_ms",
            "n_success_mean",
        ])
        for row in summary_rows:
            writer.writerow([
                row["condition"], row["users"], row["repeats"],
                f"{row['throughput_mean']:.4f}", f"{row['throughput_std']:.4f}",
                f"{row['error_rate_mean']:.4f}", f"{row['error_rate_std']:.4f}",
                f"{row['p50_mean']:.4f}", f"{row['p50_std']:.4f}",
                f"{row['p95_mean']:.4f}", f"{row['p95_std']:.4f}",
                f"{row['p99_mean']:.4f}", f"{row['p99_std']:.4f}",
                f"{row['n_success_mean']:.1f}",
            ])

    max_tp = pick_max_throughput(summary_rows, args.max_error_rate_pct)
    max_csv = output_prefix.with_name(output_prefix.name + "_max_throughput.csv")
    with max_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "users", "throughput_mean_req_s", "p95_mean_ms", "p99_mean_ms", "error_rate_mean_pct"])
        for cond in sorted(max_tp):
            row = max_tp[cond]
            writer.writerow([
                cond, row["users"], f"{row['throughput_mean']:.4f}",
                f"{row['p95_mean']:.4f}", f"{row['p99_mean']:.4f}",
                f"{row['error_rate_mean']:.4f}",
            ])

    ref_rows = {row["condition"]: row for row in summary_rows if row["users"] == args.reference_users}
    baseline_ref = ref_rows.get("baseline")

    lines = []
    lines.append("# Reviewer-Facing Overhead Results")
    lines.append("")
    lines.append(f"Reference load: **{args.reference_users} virtual users**")
    lines.append(f"Maximum-throughput criterion: **error rate <= {args.max_error_rate_pct:.1f}%**")
    lines.append("")
    lines.append("## P95/P99 Latency At Reference Load")
    lines.append("")
    lines.append("| Condition | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Error rate | P95 overhead vs baseline | P95 overhead vs LTTng only |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lttng_ref = ref_rows.get("lttng_only")
    for cond in ["baseline", "lttng_only", "lmat_async", "lmat_sync"]:
        row = ref_rows.get(cond)
        if row is None:
            continue
        if baseline_ref:
            p95_oh = ((row["p95_mean"] - baseline_ref["p95_mean"]) / baseline_ref["p95_mean"] * 100.0) if baseline_ref["p95_mean"] else float("nan")
        else:
            p95_oh = float("nan")
        if lttng_ref:
            p95_vs_lttng = ((row["p95_mean"] - lttng_ref["p95_mean"]) / lttng_ref["p95_mean"] * 100.0) if lttng_ref["p95_mean"] else float("nan")
        else:
            p95_vs_lttng = float("nan")
        lines.append(
            f"| {cond} | "
            f"{fmt(row['throughput_mean'])} ± {fmt(row['throughput_std'])} | "
            f"{fmt(row['p50_mean'])} | {fmt(row['p95_mean'])} | {fmt(row['p99_mean'])} | "
            f"{fmt(row['error_rate_mean'], 2)}% | {fmt(p95_oh, 1)}% | {fmt(p95_vs_lttng, 1)}% |"
        )

    lines.append("")
    lines.append("## Maximum Throughput")
    lines.append("")
    lines.append("| Condition | Users at max throughput | Max throughput (req/s) | P95 at max throughput (ms) | P99 at max throughput (ms) | Error rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in ["baseline", "lttng_only", "lmat_async", "lmat_sync"]:
        row = max_tp.get(cond)
        if row is None:
            continue
        lines.append(
            f"| {cond} | {row['users']} | {fmt(row['throughput_mean'])} | "
            f"{fmt(row['p95_mean'])} | {fmt(row['p99_mean'])} | {fmt(row['error_rate_mean'], 2)}% |"
        )

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Per-user summary CSV: `{detail_csv}`")
    lines.append(f"- Maximum-throughput CSV: `{max_csv}`")
    lines.append("")
    lines.append("## Interpretation Note")
    lines.append("")
    lines.append("- `lttng_only` is the direct tracing-overhead condition.")
    lines.append("- `lmat_async` is a co-located replay-based proxy for tracing-plus-inference CPU overhead, so its fairest incremental comparison is against `lttng_only`, not against a colder baseline run.")

    md_path = output_prefix.with_name(output_prefix.name + "_summary.md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote:\n- {detail_csv}\n- {max_csv}\n- {md_path}")


if __name__ == "__main__":
    main()
