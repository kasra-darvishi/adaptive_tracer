#!/usr/bin/env python3
"""Export per-sequence OOD scores for CE / anomaly-score distribution plots.

This script loads a trained checkpoint, runs the same paper-style scoring logic used
in train_sockshop.py, and writes a score CSV suitable for plot_score_distributions.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from microservice.train_sockshop import (
    _compute_mad_stats,
    _make_ood_loader,
    build_model,
    combine_paper_ood_scores,
    evaluate_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Export per-sequence LMAT scores for distribution plots")
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--load_model", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--summary_csv", default=None)
    p.add_argument("--summary_json", default=None)
    p.add_argument("--model", choices=["lstm", "transformer"], required=True)
    p.add_argument("--n_categories", type=int, required=True)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_hidden", type=int, default=1024)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", choices=["relu", "gelu", "swiglu"], default="gelu")
    p.add_argument("--tfixup", action="store_true")
    p.add_argument("--dim_sys", type=int, default=48)
    p.add_argument("--dim_entry", type=int, default=12)
    p.add_argument("--dim_ret", type=int, default=12)
    p.add_argument("--dim_proc", type=int, default=48)
    p.add_argument("--dim_pid", type=int, default=12)
    p.add_argument("--dim_tid", type=int, default=12)
    p.add_argument("--dim_order", type=int, default=12)
    p.add_argument("--dim_time", type=int, default=12)
    p.add_argument("--dim_f_mean", type=int, default=0)
    p.add_argument("--train_event_model", action="store_true")
    p.add_argument("--train_latency_model", action="store_true")
    p.add_argument("--ordinal_latency", action="store_true")
    p.add_argument("--multitask_lambda", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--chk", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def detect_splits(base: Path):
    ordered = []
    if (base / "valid_id").is_dir():
        ordered.append("valid_id")
    if (base / "test_id").is_dir():
        ordered.append("test_id")
    for prefix in ["valid_ood_", "test_ood_"]:
        for name in sorted(os.listdir(base)):
            if name.startswith(prefix) and (base / name).is_dir():
                ordered.append(name)
    return ordered


def label_for_split(split: str) -> str:
    return "normal" if split.endswith("_id") else "anomaly"


def summarize_scores(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return None
    va = a.var(ddof=1) if a.size > 1 else 0.0
    vb = b.var(ddof=1) if b.size > 1 else 0.0
    pooled = ((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1)
    if pooled <= 1e-12:
        return None
    return float((b.mean() - a.mean()) / np.sqrt(pooled))


def main():
    args = parse_args()
    print(f"[export] preprocessed_dir={args.preprocessed_dir}")
    print(f"[export] checkpoint={args.load_model}")
    print(f"[export] output_csv={args.output_csv}")

    if not (args.train_event_model or args.train_latency_model):
        raise SystemExit("At least one of --train_event_model / --train_latency_model is required")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_csv.with_name(output_csv.stem + "_summary.csv")
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_name(output_csv.stem + "_summary.json")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("[export] loading vocab ...")
    with open(Path(args.preprocessed_dir) / "vocab.pkl", "rb") as fh:
        dict_sys, dict_proc = pickle.load(fh)
    print(f"[export] vocab: {len(dict_sys)} syscalls / {len(dict_proc)} processes")

    print(f"[export] building model on {device} ...")
    model = build_model(args, len(dict_sys), len(dict_proc), device)
    print("[export] loading checkpoint ...")
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.eval()
    print("[export] checkpoint loaded")

    crit_e = nn.CrossEntropyLoss(ignore_index=0)
    crit_l = nn.BCEWithLogitsLoss() if args.ordinal_latency else nn.CrossEntropyLoss(ignore_index=0)

    base = Path(args.preprocessed_dir)
    pin = device.type == "cuda"
    split_names = detect_splits(base)
    print(f"[export] discovered splits: {split_names}")
    if not split_names:
        raise SystemExit(f"No valid/test splits found under {base}")

    split_results = {}
    for split in split_names:
        print(f"[export] evaluating split: {split}")
        loader = _make_ood_loader(str(base), split, args.batch, args.max_seq_len, pin)
        if loader is None:
            print(f"[export] skipped split (no loader): {split}")
            continue
        split_results[split] = evaluate_split(model, loader, device, args, crit_e, crit_l, return_scores=True)
        n_e = len(split_results[split].get("scores_event", []))
        n_l = len(split_results[split].get("scores_latency", []))
        print(f"[export] completed split: {split} (event_scores={n_e}, latency_scores={n_l})")

    if not split_results:
        raise SystemExit("No splits could be evaluated")

    mad_event = None
    mad_latency = None
    if args.train_event_model and args.train_latency_model and "valid_id" in split_results:
        print("[export] computing MAD stats from valid_id ...")
        mad_event = _compute_mad_stats(split_results["valid_id"]["scores_event"])
        mad_latency = _compute_mad_stats(split_results["valid_id"]["scores_latency"])
        print(f"[export] MAD event={mad_event} latency={mad_latency}")

    score_method = (
        "paper_mad_sum" if (args.train_event_model and args.train_latency_model) else
        "event_loss" if args.train_event_model else
        "latency_loss"
    )

    rows = []
    per_split_final = {}
    for split, result in split_results.items():
        score_event = np.asarray(result.get("scores_event", []), dtype=np.float64)
        score_latency = np.asarray(result.get("scores_latency", []), dtype=np.float64)
        score_final = combine_paper_ood_scores(score_event, score_latency, args, mad_event, mad_latency)
        per_split_final[split] = score_final
        label = label_for_split(split)
        n = max(score_final.size, score_event.size, score_latency.size)
        for i in range(n):
            rows.append({
                "split": split,
                "label": label,
                "score": float(score_final[i]) if i < score_final.size else "",
                "score_event": float(score_event[i]) if i < score_event.size else "",
                "score_latency": float(score_latency[i]) if i < score_latency.size else "",
                "score_method": score_method,
                "model": args.model,
                "categories": args.n_categories,
                "seq_len": args.max_seq_len,
            })

    print("[export] writing score CSV ...")
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "label", "score", "score_event", "score_latency", "score_method", "model", "categories", "seq_len"])
        writer.writeheader()
        writer.writerows(rows)

    normal_split = "valid_id" if "valid_id" in per_split_final else "test_id"
    normal_scores = per_split_final.get(normal_split, np.array([], dtype=np.float64))
    summary_rows = []
    summary_json_obj = {
        "score_method": score_method,
        "mad_event": mad_event,
        "mad_latency": mad_latency,
        "splits": {},
        "comparisons_vs_normal": {},
        "normal_reference_split": normal_split,
    }

    for split, scores in per_split_final.items():
        stats = summarize_scores(scores)
        if stats is None:
            continue
        stats["split"] = split
        stats["label"] = label_for_split(split)
        summary_rows.append(stats)
        summary_json_obj["splits"][split] = stats
        if split != normal_split and normal_scores.size > 0:
            summary_json_obj["comparisons_vs_normal"][split] = {
                "mean_gap": float(np.mean(scores) - np.mean(normal_scores)),
                "median_gap": float(np.median(scores) - np.median(normal_scores)),
                "cohen_d": cohen_d(normal_scores, scores),
            }

    print("[export] writing summary CSV ...")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "label", "count", "mean", "median", "std", "min", "max"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("[export] writing summary JSON ...")
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary_json_obj, fh, indent=2)

    print(f"Saved score CSV: {output_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")


if __name__ == "__main__":
    main()


