#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from microservice.ood_metrics import tune_threshold_max_f1
from microservice.root_cause_vectors import (
    ANOMALY_TYPES,
    MISS_LABEL,
    UNKNOWN_LABEL,
    RootCauseCalibration,
    build_cluster_prototypes,
    build_model_from_args,
    classify_record_to_prototypes,
    combine_balanced_binary_scores_labels,
    combine_paper_ood_scores,
    confusion_matrix_dict,
    extract_root_cause_records,
    make_loader,
    summarise_prototypes,
    write_predictions_csv,
)
from microservice.train_sockshop import _compute_mad_stats, evaluate_split


def get_args():
    p = argparse.ArgumentParser(description="Root-cause evaluation for LMAT SockShop models")

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
    p.add_argument("--dim_sys", type=int, default=64)
    p.add_argument("--dim_entry", type=int, default=8)
    p.add_argument("--dim_ret", type=int, default=8)
    p.add_argument("--dim_proc", type=int, default=8)
    p.add_argument("--dim_pid", type=int, default=16)
    p.add_argument("--dim_tid", type=int, default=16)
    p.add_argument("--dim_order", type=int, default=16)
    p.add_argument("--dim_time", type=int, default=16)
    p.add_argument("--dim_f_mean", type=int, default=0)
    p.add_argument("--train_event_model", action="store_true")
    p.add_argument("--train_latency_model", action="store_true")
    p.add_argument("--ordinal_latency", action="store_true")
    p.add_argument("--multitask_lambda", type=float, default=0.5)

    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--chk", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--combine_strategy", choices=["mean", "sum", "concat"], default="mean")
    p.add_argument("--centroid_source", choices=["all", "detected"], default="all")
    p.add_argument("--cluster_method", choices=["hdbscan"], default="hdbscan")
    p.add_argument("--cluster_metric", type=str, default="euclidean")
    p.add_argument("--cluster_min_size", type=int, default=128)
    p.add_argument("--cluster_min_samples", type=int, default=None)
    p.add_argument("--cluster_max_records_per_label", type=int, default=None)

    args = p.parse_args()
    if not (args.train_event_model or args.train_latency_model):
        p.error("At least one of --train_event_model / --train_latency_model is required")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str):
    print(msg, flush=True)


def fmt_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_calibration(model, args, device, crit_e, crit_l) -> tuple[RootCauseCalibration, dict]:
    pin = device.type == "cuda"
    ld_vid = make_loader(
        args.preprocessed_dir,
        "valid_id",
        batch=args.batch,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    if ld_vid is None:
        raise FileNotFoundError("valid_id split is required for root-cause calibration")

    res_vid = evaluate_split(model, ld_vid, device, args, crit_e, crit_l, return_scores=True)

    valid_ood_results = {}
    for atype in ANOMALY_TYPES:
        loader = make_loader(
            args.preprocessed_dir,
            f"valid_ood_{atype}",
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
        raise FileNotFoundError("At least one valid_ood_* split is required for root-cause evaluation")

    mad_event = None
    mad_latency = None
    if args.train_event_model and args.train_latency_model:
        mad_event = _compute_mad_stats(res_vid["scores_event"])
        mad_latency = _compute_mad_stats(res_vid["scores_latency"])

    val_id_scores = combine_paper_ood_scores(
        res_vid["scores_event"],
        res_vid["scores_latency"],
        args,
        mad_event,
        mad_latency,
    )
    val_ood_scores = []
    for atype in ANOMALY_TYPES:
        res = valid_ood_results.get(atype)
        if res is None:
            continue
        scores = combine_paper_ood_scores(
            res["scores_event"],
            res["scores_latency"],
            args,
            mad_event,
            mad_latency,
        )
        if scores.size > 0:
            val_ood_scores.append(scores)
    if not val_ood_scores:
        raise RuntimeError("No validation OOD scores available for threshold tuning")

    pooled_val_ood = np.concatenate(val_ood_scores)
    scores_val, y_val = combine_balanced_binary_scores_labels(
        val_id_scores, pooled_val_ood, seed=args.seed
    )
    if y_val.size == 0 or np.unique(y_val).size < 2:
        raise RuntimeError("Could not build a valid binary validation set for threshold tuning")
    best_t, best_f1 = tune_threshold_max_f1(scores_val, y_val, args.ood_threshold_grid)
    calibration = RootCauseCalibration(
        threshold=best_t,
        val_f1=best_f1,
        mad_event=mad_event,
        mad_latency=mad_latency,
        n_val_normal=int(val_id_scores.size),
        n_val_ood=int(pooled_val_ood.size),
    )
    return calibration, valid_ood_results


def aggregate_metrics(label_records: dict[str, list[dict]], centroid_labels: list[str]):
    all_records = [record for records in label_records.values() for record in records]

    def safe_ratio(num: int, den: int):
        return None if den == 0 else num / den

    summary = {"per_label": {}, "overall": {}}
    detected_true = []
    detected_pred = []
    e2e_true = []
    e2e_pred = []

    for label, records in label_records.items():
        total = len(records)
        detected = sum(1 for r in records if r["detected"])
        isolated_correct = sum(1 for r in records if r["isolated_correct"])
        detected_only_correct = sum(1 for r in records if r["detected_only_correct"])
        end_to_end_correct = sum(1 for r in records if r["end_to_end_correct"])

        detected_subset = [r for r in records if r["detected"]]
        summary["per_label"][label] = {
            "n_sequences": total,
            "n_detected": detected,
            "detection_recall": safe_ratio(detected, total),
            "isolated_accuracy_all": safe_ratio(isolated_correct, total),
            "detected_only_accuracy": safe_ratio(detected_only_correct, detected),
            "end_to_end_accuracy": safe_ratio(end_to_end_correct, total),
        }

        for record in detected_subset:
            detected_true.append(label)
            detected_pred.append(record["predicted_label"] or UNKNOWN_LABEL)
        for record in records:
            e2e_true.append(label)
            e2e_pred.append(record["predicted_label_e2e"])

    total = len(all_records)
    detected = sum(1 for r in all_records if r["detected"])
    isolated_correct = sum(1 for r in all_records if r["isolated_correct"])
    detected_only_correct = sum(1 for r in all_records if r["detected_only_correct"])
    end_to_end_correct = sum(1 for r in all_records if r["end_to_end_correct"])

    summary["overall"] = {
        "n_sequences": total,
        "n_detected": detected,
        "detection_recall": safe_ratio(detected, total),
        "isolated_accuracy_all": safe_ratio(isolated_correct, total),
        "detected_only_accuracy": safe_ratio(detected_only_correct, detected),
        "end_to_end_accuracy": safe_ratio(end_to_end_correct, total),
    }

    detected_labels = centroid_labels + [UNKNOWN_LABEL]
    e2e_labels = centroid_labels + [MISS_LABEL, UNKNOWN_LABEL]
    summary["confusion_detected_only"] = confusion_matrix_dict(
        detected_true, detected_pred, detected_labels
    )
    summary["confusion_end_to_end"] = confusion_matrix_dict(
        e2e_true, e2e_pred, e2e_labels
    )
    return summary


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    log(f"[RCA] device={device}")

    model, dict_sys, _, id_to_syscall = build_model_from_args(args, device)
    log(
        f"[RCA] loaded model from {args.load_model} "
        f"({len(dict_sys)} syscall tokens)"
    )

    crit_e = nn.CrossEntropyLoss(ignore_index=0) if args.train_event_model else None
    if args.train_latency_model:
        if args.ordinal_latency:
            crit_l = nn.BCEWithLogitsLoss()
        else:
            crit_l = nn.CrossEntropyLoss(ignore_index=0)
    else:
        crit_l = None

    calibration, valid_ood_results = build_calibration(model, args, device, crit_e, crit_l)
    log(
        f"[RCA] threshold={calibration.threshold:.6g} "
        f"val_f1={calibration.val_f1:.4f} "
        f"valid_id={calibration.n_val_normal} valid_ood={calibration.n_val_ood}"
    )
    if calibration.mad_event is not None and calibration.mad_latency is not None:
        log(
            "[RCA] MAD "
            f"event(median={calibration.mad_event['median']:.6g}, mad={calibration.mad_event['mad']:.6g}) "
            f"latency(median={calibration.mad_latency['median']:.6g}, mad={calibration.mad_latency['mad']:.6g})"
        )

    pin = device.type == "cuda"
    valid_records_by_label = {}
    for atype in ANOMALY_TYPES:
        loader = make_loader(
            args.preprocessed_dir,
            f"valid_ood_{atype}",
            batch=args.batch,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        if loader is None:
            continue
        records = extract_root_cause_records(
            model,
            loader,
            device,
            args,
            split_name=f"valid_ood_{atype}",
            anomaly_label=atype,
            id_to_syscall=id_to_syscall,
            score_threshold=calibration.threshold,
            mad_event=calibration.mad_event,
            mad_latency=calibration.mad_latency,
            combine_strategy=args.combine_strategy,
        )
        valid_records_by_label[atype] = records
        log(f"[RCA] valid_ood_{atype}: extracted {len(records)} sequences")

    prototypes, prototype_meta = build_cluster_prototypes(
        valid_records_by_label,
        centroid_source=args.centroid_source,
        cluster_method=args.cluster_method,
        min_cluster_size=args.cluster_min_size,
        min_samples=args.cluster_min_samples,
        cluster_metric=args.cluster_metric,
        cluster_max_records_per_label=args.cluster_max_records_per_label,
        seed=args.seed,
    )
    if not prototypes:
        raise RuntimeError("No cluster centroids could be built from validation OOD data")

    for label, meta in prototype_meta.items():
        log(
            f"[RCA] clusters {label}: total={meta['n_records_total']} "
            f"selected={meta['n_records_selected']} clustered={meta['n_records_clustered']} "
            f"vectors={meta['n_vectors_used']} n_clusters={meta['n_clusters']} "
            f"noise={meta['noise_count']}"
        )

    test_records_by_label = {}
    flat_predictions = []
    centroid_labels = sorted({proto.label for proto in prototypes.values()})
    for atype in ANOMALY_TYPES:
        loader = make_loader(
            args.preprocessed_dir,
            f"test_ood_{atype}",
            batch=args.batch,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        if loader is None:
            continue
        records = extract_root_cause_records(
            model,
            loader,
            device,
            args,
            split_name=f"test_ood_{atype}",
            anomaly_label=atype,
            id_to_syscall=id_to_syscall,
            score_threshold=calibration.threshold,
            mad_event=calibration.mad_event,
            mad_latency=calibration.mad_latency,
            combine_strategy=args.combine_strategy,
        )
        for record in records:
            pred_label, pred_cluster_key, pred_sim = classify_record_to_prototypes(
                record, prototypes
            )
            record["predicted_label"] = pred_label
            record["predicted_cluster_key"] = pred_cluster_key
            record["predicted_similarity"] = pred_sim
            record["isolated_correct"] = bool(pred_label == atype)
            record["detected_only_correct"] = bool(record["detected"] and pred_label == atype)
            if record["detected"]:
                record["predicted_label_e2e"] = pred_label if pred_label is not None else UNKNOWN_LABEL
            else:
                record["predicted_label_e2e"] = MISS_LABEL
            record["end_to_end_correct"] = bool(record["predicted_label_e2e"] == atype)
        test_records_by_label[atype] = records
        flat_predictions.extend(records)
        log(f"[RCA] test_ood_{atype}: extracted {len(records)} sequences")

    metrics = aggregate_metrics(test_records_by_label, centroid_labels)

    centroids_json = summarise_prototypes(
        prototypes, prototype_meta, id_to_syscall, top_k=15
    )

    results = {
        "config": {
            "preprocessed_dir": args.preprocessed_dir,
            "load_model": args.load_model,
            "model": args.model,
            "combine_strategy": args.combine_strategy,
            "centroid_source": args.centroid_source,
            "cluster_method": args.cluster_method,
            "cluster_metric": args.cluster_metric,
            "cluster_min_size": args.cluster_min_size,
            "cluster_min_samples": args.cluster_min_samples,
            "cluster_max_records_per_label": args.cluster_max_records_per_label,
            "max_samples": args.max_samples,
            "train_event_model": args.train_event_model,
            "train_latency_model": args.train_latency_model,
        },
        "calibration": {
            "threshold": calibration.threshold,
            "val_f1": calibration.val_f1,
            "n_val_normal": calibration.n_val_normal,
            "n_val_ood": calibration.n_val_ood,
            "mad_event": calibration.mad_event,
            "mad_latency": calibration.mad_latency,
        },
        "centroids": centroids_json,
        "metrics": metrics,
    }

    results_path = os.path.join(args.output_dir, "root_cause_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    predictions_path = os.path.join(args.output_dir, "root_cause_predictions.csv")
    write_predictions_csv(predictions_path, flat_predictions)

    centroids_npz = os.path.join(args.output_dir, "root_cause_centroids.npz")
    np.savez(centroids_npz, **{proto.key: proto.centroid for proto in prototypes.values()})

    log(f"[RCA] results -> {results_path}")
    log(f"[RCA] predictions -> {predictions_path}")
    log(f"[RCA] centroids -> {centroids_npz}")
    overall = metrics["overall"]
    log(
        f"[RCA] overall detection_recall={fmt_metric(overall['detection_recall'])} "
        f"detected_only_acc={fmt_metric(overall['detected_only_accuracy'])} "
        f"end_to_end_acc={fmt_metric(overall['end_to_end_accuracy'])}"
    )


if __name__ == "__main__":
    main()
