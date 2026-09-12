#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from microservice.ood_metrics import tune_threshold_max_f1
from microservice.root_cause_vectors import ANOMALY_TYPES, build_model_from_args, make_loader
from microservice.train_sockshop import (
    _compute_mad_stats,
    combine_full_binary_scores_labels,
    combine_paper_ood_scores,
    evaluate_split,
    forward_batch,
    per_sequence_event_ce,
    per_sequence_latency_ce,
    _scores_to_numpy,
)


def get_args():
    p = argparse.ArgumentParser(
        description="Replay adaptive tracing decisions using paper-style LMAT anomaly scores"
    )

    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--load_model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_categories", type=int, default=6)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--ood_threshold_grid", type=int, default=100)

    p.add_argument("--model", choices=["lstm", "transformer"], default="transformer")
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
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--progress_every_batches", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--chk", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--trigger_ratio", type=float, default=0.8)
    p.add_argument("--warmup_sequences", type=int, default=256)
    p.add_argument("--linger_sequences", type=int, default=16)
    p.add_argument("--normal_split", default="test_id")
    p.add_argument(
        "--threshold_mode",
        choices=["f1", "normal_quantile", "f1_plus_margin"],
        default="normal_quantile",
    )
    p.add_argument(
        "--normal_quantile",
        type=float,
        default=0.9995,
        help="Used when --threshold_mode=normal_quantile; threshold is the given valid_id score quantile.",
    )
    p.add_argument(
        "--threshold_margin",
        type=float,
        default=1.0,
        help="Used when --threshold_mode=f1_plus_margin; adds this value to the F1-tuned threshold.",
    )
    p.add_argument(
        "--scenario_mode",
        choices=["abrupt_burst", "pure_normal", "both"],
        default="both",
    )

    args = p.parse_args()
    if not (args.train_event_model or args.train_latency_model):
        p.error("At least one of --train_event_model / --train_latency_model is required")
    if args.window_size <= 0:
        p.error("--window_size must be > 0")
    if not (0.0 < args.trigger_ratio <= 1.0):
        p.error("--trigger_ratio must be in (0, 1]")
    if args.linger_sequences < 0:
        p.error("--linger_sequences must be >= 0")
    if args.warmup_sequences < 0:
        p.error("--warmup_sequences must be >= 0")
    if args.progress_every_batches <= 0:
        p.error("--progress_every_batches must be > 0")
    if not (0.0 < args.normal_quantile < 1.0):
        p.error("--normal_quantile must be in (0, 1)")
    return args


def log(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_calibration(model, args, device, crit_e, crit_l):
    pin = device.type == "cuda"
    valid_id_loader = make_loader(
        args.preprocessed_dir,
        "valid_id",
        batch=args.batch,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    if valid_id_loader is None:
        raise FileNotFoundError("valid_id split is required for adaptive replay calibration")

    res_vid = evaluate_split(model, valid_id_loader, device, args, crit_e, crit_l, return_scores=True)

    valid_ood_results = {}
    for atype in ANOMALY_TYPES:
        split_name = f"valid_ood_{atype}"
        loader = make_loader(
            args.preprocessed_dir,
            split_name,
            batch=args.batch,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        if loader is None:
            continue
        valid_ood_results[atype] = evaluate_split(
            model, loader, device, args, crit_e, crit_l, return_scores=True
        )
    if not valid_ood_results:
        raise FileNotFoundError("At least one valid_ood_* split is required for adaptive replay")

    mad_event = None
    mad_latency = None
    if args.train_event_model and args.train_latency_model:
        valid_event_pool = [res_vid["scores_event"]]
        valid_latency_pool = [res_vid["scores_latency"]]
        for atype in ANOMALY_TYPES:
            res_vood = valid_ood_results.get(atype)
            if res_vood is None:
                continue
            if res_vood["scores_event"].size > 0:
                valid_event_pool.append(res_vood["scores_event"])
            if res_vood["scores_latency"].size > 0:
                valid_latency_pool.append(res_vood["scores_latency"])
        mad_event = _compute_mad_stats(np.concatenate(valid_event_pool)) if valid_event_pool else None
        mad_latency = _compute_mad_stats(np.concatenate(valid_latency_pool)) if valid_latency_pool else None

    val_id_scores = combine_paper_ood_scores(
        res_vid["scores_event"], res_vid["scores_latency"], args, mad_event, mad_latency
    )
    val_ood_scores_all = []
    for atype in ANOMALY_TYPES:
        res_vood = valid_ood_results.get(atype)
        if res_vood is None:
            continue
        s_vood = combine_paper_ood_scores(
            res_vood["scores_event"], res_vood["scores_latency"], args, mad_event, mad_latency
        )
        if s_vood.size > 0:
            val_ood_scores_all.append(s_vood)
    if not val_ood_scores_all:
        raise RuntimeError("No validation OOD scores available for threshold tuning")
    pooled_val_ood = np.concatenate(val_ood_scores_all)
    scores_val, y_val = combine_full_binary_scores_labels(val_id_scores, pooled_val_ood)
    if y_val.size == 0 or np.unique(y_val).size < 2:
        raise RuntimeError("Could not build a valid binary validation set for threshold tuning")
    threshold_f1, best_f1 = tune_threshold_max_f1(scores_val, y_val, args.ood_threshold_grid)
    threshold_quantile = float(np.quantile(val_id_scores, args.normal_quantile)) if val_id_scores.size > 0 else float("nan")
    if args.threshold_mode == "f1":
        threshold = threshold_f1
    elif args.threshold_mode == "normal_quantile":
        threshold = threshold_quantile
    else:
        threshold = float(threshold_f1 + args.threshold_margin)
    return {
        "threshold": float(threshold),
        "threshold_f1": float(threshold_f1),
        "threshold_normal_quantile": float(threshold_quantile),
        "val_f1": float(best_f1),
        "n_val_normal": int(val_id_scores.size),
        "n_val_ood": int(pooled_val_ood.size),
        "mad_event": mad_event,
        "mad_latency": mad_latency,
        "threshold_mode": args.threshold_mode,
        "normal_quantile": float(args.normal_quantile),
        "threshold_margin": float(args.threshold_margin),
    }


@torch.no_grad()
def extract_stream_scores(model, loader, device, args, split_name, mad_event, mad_latency):
    if loader is None:
        return None

    scores = []
    scores_event = []
    scores_latency = []
    seq_len = []
    req_dur_ms = []
    is_anomaly = []

    model.eval()
    batch_count = 0
    sequence_count = 0
    start_t = time.perf_counter()
    log(f"[ATR] {split_name}: starting score extraction")
    for batch in loader:
        batch_count += 1
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_e, logits_l = forward_batch(model, batch, device, args)

        tgt_call = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
        tgt_lat = batch["tgt_lat"].to(device, dtype=torch.long, non_blocking=True)
        seq_sc_e = per_sequence_event_ce(logits_e, tgt_call) if logits_e.numel() > 0 else None
        seq_sc_l = (
            per_sequence_latency_ce(logits_l, tgt_lat, args.ordinal_latency)
            if logits_l.numel() > 0
            else None
        )
        arr_e = _scores_to_numpy(seq_sc_e)
        arr_l = _scores_to_numpy(seq_sc_l)
        arr_score = combine_paper_ood_scores(arr_e, arr_l, args, mad_event, mad_latency)

        scores.append(arr_score)
        scores_event.append(arr_e)
        scores_latency.append(arr_l)
        seq_len_batch = batch["seq_len"].detach().cpu().numpy().astype(np.int32, copy=False)
        seq_len.append(seq_len_batch)
        req_dur_ms.append(batch["req_dur_ms"].detach().cpu().numpy().astype(np.float64, copy=False))
        is_anomaly.append(batch["is_anomaly"].detach().cpu().numpy().astype(np.int32, copy=False))
        sequence_count += int(seq_len_batch.shape[0])

        if batch_count % args.progress_every_batches == 0:
            elapsed = time.perf_counter() - start_t
            msg = (
                f"[ATR] {split_name}: processed {batch_count} batches | "
                f"{sequence_count} sequences | elapsed={elapsed:.1f}s"
            )
            if elapsed > 0:
                msg += f" | {sequence_count / elapsed:.1f} seq/s"
            log(msg)

    wall_time_s = time.perf_counter() - start_t

    if not scores:
        return None
    score_arr = np.concatenate(scores).astype(np.float64, copy=False)
    event_arr = np.concatenate(scores_event).astype(np.float64, copy=False) if scores_event else np.array([], dtype=np.float64)
    latency_arr = np.concatenate(scores_latency).astype(np.float64, copy=False) if scores_latency else np.array([], dtype=np.float64)
    seq_len_arr = np.concatenate(seq_len).astype(np.int32, copy=False)
    req_dur_arr = np.concatenate(req_dur_ms).astype(np.float64, copy=False)
    anomaly_arr = np.concatenate(is_anomaly).astype(np.int32, copy=False)
    n_seq = int(score_arr.size)
    stream_time_ms = float(np.sum(req_dur_arr))

    return {
        "split": split_name,
        "scores": score_arr,
        "scores_event": event_arr,
        "scores_latency": latency_arr,
        "seq_len": seq_len_arr,
        "req_dur_ms": req_dur_arr,
        "is_anomaly": anomaly_arr,
        "n_sequences": n_seq,
        "n_batches": int(batch_count),
        "wall_time_s": float(wall_time_s),
        "ms_per_sequence": float((wall_time_s * 1000.0) / n_seq) if n_seq > 0 else None,
        "sequences_per_s": float(n_seq / wall_time_s) if wall_time_s > 0 else None,
        "stream_time_ms": stream_time_ms,
        "realtime_ratio": float((wall_time_s * 1000.0) / stream_time_ms) if stream_time_ms > 0 else None,
    }


def concat_streams(parts, split_name):
    parts = [p for p in parts if p is not None and p["n_sequences"] > 0]
    if not parts:
        return None
    total_wall = float(sum(p["wall_time_s"] for p in parts))
    batch_counts = [p.get("n_batches") for p in parts if p.get("n_batches") is not None]
    scores = np.concatenate([p["scores"] for p in parts]).astype(np.float64, copy=False)
    scores_event = np.concatenate([p["scores_event"] for p in parts]).astype(np.float64, copy=False)
    scores_latency = np.concatenate([p["scores_latency"] for p in parts]).astype(np.float64, copy=False)
    seq_len = np.concatenate([p["seq_len"] for p in parts]).astype(np.int32, copy=False)
    req_dur_ms = np.concatenate([p["req_dur_ms"] for p in parts]).astype(np.float64, copy=False)
    is_anomaly = np.concatenate([p["is_anomaly"] for p in parts]).astype(np.int32, copy=False)
    n_seq = int(scores.size)
    stream_time_ms = float(np.sum(req_dur_ms))
    return {
        "split": split_name,
        "scores": scores,
        "scores_event": scores_event,
        "scores_latency": scores_latency,
        "seq_len": seq_len,
        "req_dur_ms": req_dur_ms,
        "is_anomaly": is_anomaly,
        "n_sequences": n_seq,
        "n_batches": int(sum(batch_counts)) if batch_counts else None,
        "wall_time_s": total_wall,
        "ms_per_sequence": float((total_wall * 1000.0) / n_seq) if n_seq > 0 else None,
        "sequences_per_s": float(n_seq / total_wall) if total_wall > 0 else None,
        "stream_time_ms": stream_time_ms,
        "realtime_ratio": float((total_wall * 1000.0) / stream_time_ms) if stream_time_ms > 0 else None,
    }


def simulate_controller(stream, threshold, window_size, trigger_ratio, linger_sequences, onset_index=0):
    scores = np.asarray(stream["scores"], dtype=np.float64)
    seq_len = np.asarray(stream["seq_len"], dtype=np.int64)
    req_dur_ms = np.asarray(stream["req_dur_ms"], dtype=np.float64)
    is_anomaly = np.asarray(stream["is_anomaly"], dtype=np.int64)

    n = int(scores.size)
    if n == 0:
        return {
            "n_sequences": 0,
            "trigger_count": 0,
            "triggered_after_onset": False,
        }

    trigger_count_needed = max(1, int(math.ceil(window_size * trigger_ratio)))
    detected = scores > threshold
    recorded = np.zeros(n, dtype=np.bool_)
    recent = deque(maxlen=window_size)
    trigger_indices = []
    capture_active = False
    activate_next = False
    linger_remaining = 0

    controller_start = time.perf_counter()
    for idx in range(n):
        if activate_next:
            capture_active = True
            linger_remaining = max(int(linger_sequences), 0)
            activate_next = False

        recorded[idx] = capture_active
        is_detected = bool(detected[idx])
        recent.append(1 if is_detected else 0)

        if capture_active:
            if is_detected:
                linger_remaining = max(int(linger_sequences), 0)
            elif linger_remaining > 0:
                linger_remaining -= 1
            else:
                capture_active = False

        if (not capture_active) and len(recent) == window_size and sum(recent) >= trigger_count_needed:
            trigger_indices.append(idx)
            activate_next = True

    controller_wall_s = time.perf_counter() - controller_start

    onset_index = int(max(0, min(onset_index, n)))
    post_onset_triggers = [idx for idx in trigger_indices if idx >= onset_index]
    first_trigger_idx = post_onset_triggers[0] if post_onset_triggers else None
    capture_start_idx = min(n, first_trigger_idx + 1) if first_trigger_idx is not None else n

    anomaly_mask = is_anomaly.astype(bool)
    anomaly_from_onset = np.zeros(n, dtype=np.bool_)
    anomaly_from_onset[onset_index:] = anomaly_mask[onset_index:]
    missed_pretrigger = anomaly_from_onset & (~recorded) & (np.arange(n) < capture_start_idx)

    total_anomaly_sequences = int(anomaly_from_onset.sum())
    total_anomaly_events = int(seq_len[anomaly_from_onset].sum()) if total_anomaly_sequences > 0 else 0
    total_anomaly_duration_ms = float(req_dur_ms[anomaly_from_onset].sum()) if total_anomaly_sequences > 0 else 0.0

    missed_anomaly_sequences = int(missed_pretrigger.sum())
    missed_anomaly_events = int(seq_len[missed_pretrigger].sum()) if missed_anomaly_sequences > 0 else 0
    missed_anomaly_duration_ms = float(req_dur_ms[missed_pretrigger].sum()) if missed_anomaly_sequences > 0 else 0.0

    recorded_sequences = int(recorded.sum())
    recorded_events = int(seq_len[recorded].sum()) if recorded_sequences > 0 else 0
    recorded_duration_ms = float(req_dur_ms[recorded].sum()) if recorded_sequences > 0 else 0.0

    total_sequences = n
    total_events = int(seq_len.sum())
    total_duration_ms = float(req_dur_ms.sum())

    false_trigger_count = int(sum(1 for idx in trigger_indices if idx < onset_index))
    trigger_delay_ms = None
    trigger_delay_sequences = None
    if first_trigger_idx is not None:
        trigger_delay_ms = float(req_dur_ms[onset_index : first_trigger_idx + 1].sum())
        trigger_delay_sequences = int(first_trigger_idx - onset_index + 1)

    return {
        "n_sequences": total_sequences,
        "n_anomaly_sequences": total_anomaly_sequences,
        "n_recorded_sequences": recorded_sequences,
        "n_recorded_events": recorded_events,
        "n_recorded_duration_ms": recorded_duration_ms,
        "recorded_sequence_fraction": (recorded_sequences / total_sequences) if total_sequences > 0 else None,
        "recorded_event_fraction": (recorded_events / total_events) if total_events > 0 else None,
        "recorded_duration_fraction": (recorded_duration_ms / total_duration_ms) if total_duration_ms > 0 else None,
        "trace_reduction_sequences": 1.0 - (recorded_sequences / total_sequences) if total_sequences > 0 else None,
        "trace_reduction_events": 1.0 - (recorded_events / total_events) if total_events > 0 else None,
        "trace_reduction_duration_ms": 1.0 - (recorded_duration_ms / total_duration_ms) if total_duration_ms > 0 else None,
        "trigger_count": int(len(trigger_indices)),
        "triggered_after_onset": first_trigger_idx is not None,
        "first_trigger_index": first_trigger_idx,
        "capture_start_index": capture_start_idx if first_trigger_idx is not None else None,
        "trigger_delay_ms": trigger_delay_ms,
        "trigger_delay_sequences": trigger_delay_sequences,
        "missed_anomaly_sequences": missed_anomaly_sequences,
        "missed_anomaly_events": missed_anomaly_events,
        "missed_anomaly_duration_ms": missed_anomaly_duration_ms,
        "miss_rate_sequences": (missed_anomaly_sequences / total_anomaly_sequences) if total_anomaly_sequences > 0 else None,
        "miss_rate_events": (missed_anomaly_events / total_anomaly_events) if total_anomaly_events > 0 else None,
        "miss_rate_duration_ms": (missed_anomaly_duration_ms / total_anomaly_duration_ms) if total_anomaly_duration_ms > 0 else None,
        "anomaly_capture_recall_sequences": 1.0 - (missed_anomaly_sequences / total_anomaly_sequences) if total_anomaly_sequences > 0 else None,
        "anomaly_capture_recall_events": 1.0 - (missed_anomaly_events / total_anomaly_events) if total_anomaly_events > 0 else None,
        "false_trigger_count_before_onset": false_trigger_count,
        "false_trigger_rate_per_1k_normal_sequences": (1000.0 * false_trigger_count / max(onset_index, 1)),
        "controller_wall_s": float(controller_wall_s),
        "controller_us_per_sequence": float((controller_wall_s * 1e6) / total_sequences) if total_sequences > 0 else None,
    }


def normal_only_metrics(stream, threshold, window_size, trigger_ratio, linger_sequences):
    metrics = simulate_controller(
        stream,
        threshold=threshold,
        window_size=window_size,
        trigger_ratio=trigger_ratio,
        linger_sequences=linger_sequences,
        onset_index=len(stream["scores"]),
    )
    metrics["false_trigger_count"] = metrics.pop("trigger_count", 0)
    metrics["false_trigger_rate_per_1k_sequences"] = (
        1000.0 * metrics["false_trigger_count"] / max(metrics["n_sequences"], 1)
    )
    return metrics


def scenario_csv_rows(scenarios):
    rows = []
    for scenario_name, metrics in scenarios.items():
        rows.append(
            {
                "scenario": scenario_name,
                "triggered_after_onset": metrics.get("triggered_after_onset"),
                "trigger_delay_ms": metrics.get("trigger_delay_ms"),
                "trigger_delay_sequences": metrics.get("trigger_delay_sequences"),
                "miss_rate_sequences": metrics.get("miss_rate_sequences"),
                "miss_rate_events": metrics.get("miss_rate_events"),
                "false_trigger_count_before_onset": metrics.get("false_trigger_count_before_onset"),
                "trace_reduction_events": metrics.get("trace_reduction_events"),
                "trace_reduction_sequences": metrics.get("trace_reduction_sequences"),
                "recorded_event_fraction": metrics.get("recorded_event_fraction"),
                "controller_us_per_sequence": metrics.get("controller_us_per_sequence"),
            }
        )
    return rows


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log(f"[ATR] device={device}")

    model, dict_sys, dict_proc, _ = build_model_from_args(args, device)
    log(f"[ATR] loaded model from {args.load_model} ({len(dict_sys)} syscall tokens)")

    crit_e = nn.CrossEntropyLoss(ignore_index=0)
    crit_l = nn.BCEWithLogitsLoss() if args.ordinal_latency else nn.CrossEntropyLoss(ignore_index=0)

    calibration = build_calibration(model, args, device, crit_e, crit_l)
    log(
        f"[ATR] threshold={calibration['threshold']:.6f} "
        f"(mode={calibration['threshold_mode']}) "
        f"val_f1={calibration['val_f1']:.4f} "
        f"valid_id={calibration['n_val_normal']} valid_ood={calibration['n_val_ood']}"
    )
    log(
        f"[ATR] threshold candidates: "
        f"f1={calibration['threshold_f1']:.6f} "
        f"normal_q({calibration['normal_quantile']:.4f})={calibration['threshold_normal_quantile']:.6f}"
    )
    if calibration["threshold_mode"] == "f1_plus_margin":
        log(f"[ATR] threshold margin applied: +{calibration['threshold_margin']:.6f}")
    if calibration["mad_event"] is not None and calibration["mad_latency"] is not None:
        log(
            f"[ATR] MAD event(median={calibration['mad_event']['median']:.6g}, "
            f"mad={calibration['mad_event']['mad']:.6g}) "
            f"latency(median={calibration['mad_latency']['median']:.6g}, "
            f"mad={calibration['mad_latency']['mad']:.6g})"
        )

    pin = device.type == "cuda"
    extracted = {}
    normal_loader = make_loader(
        args.preprocessed_dir,
        args.normal_split,
        batch=args.batch,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    if normal_loader is None:
        raise FileNotFoundError(f"Normal replay split not found: {args.normal_split}")

    extracted[args.normal_split] = extract_stream_scores(
        model,
        normal_loader,
        device,
        args,
        args.normal_split,
        calibration["mad_event"],
        calibration["mad_latency"],
    )
    log(
        f"[ATR] {args.normal_split}: scored {extracted[args.normal_split]['n_sequences']} sequences "
        f"in {extracted[args.normal_split]['wall_time_s']:.2f}s "
        f"({extracted[args.normal_split]['ms_per_sequence']:.4f} ms/seq)"
    )

    for atype in ANOMALY_TYPES:
        split_name = f"test_ood_{atype}"
        loader = make_loader(
            args.preprocessed_dir,
            split_name,
            batch=args.batch,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        if loader is None:
            continue
        extracted[split_name] = extract_stream_scores(
            model,
            loader,
            device,
            args,
            split_name,
            calibration["mad_event"],
            calibration["mad_latency"],
        )
        log(
            f"[ATR] {split_name}: scored {extracted[split_name]['n_sequences']} sequences "
            f"in {extracted[split_name]['wall_time_s']:.2f}s "
            f"({extracted[split_name]['ms_per_sequence']:.4f} ms/seq)"
        )

    scenarios = {}
    normal_stream = extracted[args.normal_split]

    if args.scenario_mode in ("pure_normal", "both"):
        log(
            f"[ATR] pure_normal: simulating controller "
            f"(window={args.window_size}, ratio={args.trigger_ratio:.3f}, linger={args.linger_sequences})"
        )
        metrics = normal_only_metrics(
            normal_stream,
            threshold=calibration["threshold"],
            window_size=args.window_size,
            trigger_ratio=args.trigger_ratio,
            linger_sequences=args.linger_sequences,
        )
        scenarios["pure_normal"] = metrics
        log(
            f"[ATR] pure_normal false_triggers={metrics['false_trigger_count']} "
            f"trace_reduction_events={metrics['trace_reduction_events']:.4f}"
        )

    abrupt_metrics = []
    if args.scenario_mode in ("abrupt_burst", "both"):
        warmup_n = min(args.warmup_sequences, normal_stream["n_sequences"])
        normal_prefix = {
            k: (v[:warmup_n] if isinstance(v, np.ndarray) else v)
            for k, v in normal_stream.items()
        }
        normal_prefix["n_sequences"] = warmup_n
        normal_prefix["wall_time_s"] = float(normal_stream["wall_time_s"] * (warmup_n / max(normal_stream["n_sequences"], 1)))
        normal_prefix["n_batches"] = None
        normal_prefix["ms_per_sequence"] = normal_stream["ms_per_sequence"]
        normal_prefix["sequences_per_s"] = normal_stream["sequences_per_s"]
        normal_prefix["stream_time_ms"] = float(np.sum(normal_prefix["req_dur_ms"])) if warmup_n > 0 else 0.0
        normal_prefix["realtime_ratio"] = (
            float((normal_prefix["wall_time_s"] * 1000.0) / normal_prefix["stream_time_ms"])
            if normal_prefix["stream_time_ms"] > 0
            else None
        )

        for atype in ANOMALY_TYPES:
            split_name = f"test_ood_{atype}"
            if split_name not in extracted:
                continue
            scenario_name = f"{atype}_abrupt_burst"
            log(
                f"[ATR] {scenario_name}: simulating controller "
                f"(warmup={warmup_n}, window={args.window_size}, "
                f"ratio={args.trigger_ratio:.3f}, linger={args.linger_sequences})"
            )
            scenario_stream = concat_streams([normal_prefix, extracted[split_name]], scenario_name)
            metrics = simulate_controller(
                scenario_stream,
                threshold=calibration["threshold"],
                window_size=args.window_size,
                trigger_ratio=args.trigger_ratio,
                linger_sequences=args.linger_sequences,
                onset_index=warmup_n,
            )
            metrics["inference_wall_s"] = float(scenario_stream["wall_time_s"])
            metrics["inference_ms_per_sequence"] = scenario_stream["ms_per_sequence"]
            metrics["inference_sequences_per_s"] = scenario_stream["sequences_per_s"]
            metrics["stream_time_ms"] = scenario_stream["stream_time_ms"]
            metrics["realtime_ratio_inference_only"] = scenario_stream["realtime_ratio"]
            processing_ratio = None
            if scenario_stream["stream_time_ms"] > 0:
                processing_ratio = float(
                    ((scenario_stream["wall_time_s"] + metrics["controller_wall_s"]) * 1000.0)
                    / scenario_stream["stream_time_ms"]
                )
            metrics["realtime_ratio_with_controller"] = processing_ratio
            scenarios[scenario_name] = metrics
            abrupt_metrics.append(metrics)
            log(
                f"[ATR] {scenario_name}: triggered={metrics['triggered_after_onset']} "
                f"delay_ms={metrics['trigger_delay_ms']} "
                f"miss_rate_events={metrics['miss_rate_events']:.4f} "
                f"trace_reduction_events={metrics['trace_reduction_events']:.4f}"
            )

    aggregate = {}
    if abrupt_metrics:
        for key in [
            "trigger_delay_ms",
            "trigger_delay_sequences",
            "miss_rate_sequences",
            "miss_rate_events",
            "trace_reduction_events",
            "trace_reduction_sequences",
            "anomaly_capture_recall_events",
            "realtime_ratio_with_controller",
        ]:
            vals = [m[key] for m in abrupt_metrics if m.get(key) is not None and np.isfinite(m.get(key))]
            aggregate[key] = float(np.mean(vals)) if vals else None
        aggregate["n_scenarios"] = len(abrupt_metrics)

    results = {
        "config": {
            "preprocessed_dir": args.preprocessed_dir,
            "load_model": args.load_model,
            "model": args.model,
            "window_size": args.window_size,
            "trigger_ratio": args.trigger_ratio,
            "warmup_sequences": args.warmup_sequences,
            "linger_sequences": args.linger_sequences,
            "normal_split": args.normal_split,
            "scenario_mode": args.scenario_mode,
            "train_event_model": args.train_event_model,
            "train_latency_model": args.train_latency_model,
            "max_samples": args.max_samples,
        },
        "calibration": calibration,
        "extraction": {
            split: {
                "n_sequences": data["n_sequences"],
                "wall_time_s": data["wall_time_s"],
                "ms_per_sequence": data["ms_per_sequence"],
                "sequences_per_s": data["sequences_per_s"],
                "stream_time_ms": data["stream_time_ms"],
                "realtime_ratio": data["realtime_ratio"],
            }
            for split, data in extracted.items()
        },
        "scenarios": scenarios,
        "aggregate": aggregate,
    }

    results_path = os.path.join(args.output_dir, "adaptive_trace_replay_results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    csv_path = os.path.join(args.output_dir, "adaptive_trace_replay_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scenario",
                "triggered_after_onset",
                "trigger_delay_ms",
                "trigger_delay_sequences",
                "miss_rate_sequences",
                "miss_rate_events",
                "false_trigger_count_before_onset",
                "trace_reduction_events",
                "trace_reduction_sequences",
                "recorded_event_fraction",
                "controller_us_per_sequence",
            ],
        )
        writer.writeheader()
        writer.writerows(scenario_csv_rows(scenarios))

    log(f"[ATR] results -> {results_path}")
    log(f"[ATR] summary -> {csv_path}")
    if aggregate:
        log(
            f"[ATR] abrupt mean delay_ms={aggregate.get('trigger_delay_ms')} "
            f"miss_rate_events={aggregate.get('miss_rate_events')} "
            f"trace_reduction_events={aggregate.get('trace_reduction_events')}"
        )


if __name__ == "__main__":
    main()
