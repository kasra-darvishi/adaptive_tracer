#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from validate_lmat_shard import (
    Dictionary,
    fail,
    info,
    load_delay_spans,
    load_vocab,
    validate_delay_spans,
    validate_shard,
)


EXPECTED_SPLITS = [
    "train_id",
    "valid_id",
    "test_id",
    "valid_ood_cpu",
    "test_ood_cpu",
    "valid_ood_disk",
    "test_ood_disk",
    "valid_ood_mem",
    "test_ood_mem",
    "valid_ood_net",
    "test_ood_net",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a full LMAT preprocessed dataset directory."
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument("--strict_manifest", action="store_true")
    p.add_argument("--show_split_stats", action="store_true")
    return p.parse_args()


def split_expected_anomaly(split_name: str) -> int:
    return 0 if split_name.endswith("_id") else 1


def load_json_if_exists(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_split(
    split_dir: str,
    split_name: str,
    dict_sys: Dictionary,
    dict_proc: Dictionary,
    delay_spans: dict,
):
    meta_path = os.path.join(split_dir, "meta.json")
    meta = load_json_if_exists(meta_path)
    shard_files = sorted(
        name for name in os.listdir(split_dir)
        if name.startswith("shard_") and name.endswith(".npz")
    )
    if not shard_files:
        fail(f"{split_name}: no shard_*.npz files found")

    expected_anomaly = split_expected_anomaly(split_name)
    total_sequences = 0
    total_truncated = 0
    min_seq_len = None
    max_seq_len = 0
    max_shard_width = 0
    lat_hist = Counter()

    for shard_name in shard_files:
        shard_path = os.path.join(split_dir, shard_name)
        arrays = validate_shard(
            shard_path,
            dict_sys,
            dict_proc,
            delay_spans,
            expected_anomaly,
        )
        seq_len = arrays["seq_len"]
        total_sequences += int(seq_len.shape[0])
        min_seq_len = int(np.min(seq_len)) if min_seq_len is None else min(min_seq_len, int(np.min(seq_len)))
        max_seq_len = max(max_seq_len, int(np.max(seq_len)))
        max_shard_width = max(max_shard_width, int(arrays["call"].shape[1]))
        last_tokens = arrays["call"][np.arange(seq_len.shape[0]), seq_len - 1]
        total_truncated += int(np.sum(last_tokens == 4))
        lat_hist.update(np.asarray(arrays["lat_cat"]).ravel().tolist())

    if meta is not None:
        if int(meta.get("n_sequences", -1)) != total_sequences:
            fail(f"{split_name}: meta.json n_sequences mismatch ({meta.get('n_sequences')} != {total_sequences})")
        if int(meta.get("n_shards", -1)) != len(shard_files):
            fail(f"{split_name}: meta.json n_shards mismatch ({meta.get('n_shards')} != {len(shard_files)})")
        if int(meta.get("is_anomaly", -1)) != expected_anomaly:
            fail(f"{split_name}: meta.json is_anomaly mismatch")

    return {
        "split": split_name,
        "n_shards": len(shard_files),
        "n_sequences": total_sequences,
        "is_anomaly": expected_anomaly,
        "min_seq_len": min_seq_len or 0,
        "max_seq_len": max_seq_len,
        "max_shard_width": max_shard_width,
        "truncated_sequences": total_truncated,
        "truncation_rate": (total_truncated / total_sequences) if total_sequences else 0.0,
        "lat_cat_hist": dict(sorted((int(k), int(v)) for k, v in lat_hist.items())),
    }


def validate_manifest(manifest: dict | None, split_summaries: dict, dict_sys: Dictionary, dict_proc: Dictionary, delay_spans: dict, strict: bool):
    if manifest is None:
        info("dataset_manifest.json not found; skipping manifest cross-checks.")
        return

    vocab = manifest.get("vocab", {})
    if vocab:
        if int(vocab.get("n_syscall", -1)) != len(dict_sys):
            fail("Manifest n_syscall does not match vocab.pkl")
        if int(vocab.get("n_process", -1)) != len(dict_proc):
            fail("Manifest n_process does not match vocab.pkl")
        if int(vocab.get("n_delay_events", -1)) != len(delay_spans):
            fail("Manifest n_delay_events does not match delay_spans.pkl")

    manifest_splits = {item["name"] for item in manifest.get("splits", []) if "name" in item}
    for split_name in split_summaries:
        if strict and split_name not in manifest_splits:
            fail(f"Manifest is missing split {split_name}")

    manifest_summaries = manifest.get("split_summaries", {})
    for split_name, stats in split_summaries.items():
        if split_name not in manifest_summaries:
            if strict:
                fail(f"Manifest split_summaries missing {split_name}")
            continue
        run_summaries = manifest_summaries[split_name]
        seq_total = sum(int(item.get("n_sequences", 0)) for item in run_summaries)
        if seq_total != stats["n_sequences"]:
            fail(f"Manifest split_summaries total mismatch for {split_name} ({seq_total} != {stats['n_sequences']})")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    vocab_path = os.path.join(data_dir, "vocab.pkl")
    delay_path = os.path.join(data_dir, "delay_spans.pkl")
    manifest_path = os.path.join(data_dir, "dataset_manifest.json")

    if not os.path.isfile(vocab_path):
        fail(f"Missing file: {vocab_path}")
    if not os.path.isfile(delay_path):
        fail(f"Missing file: {delay_path}")

    dict_sys, dict_proc = load_vocab(vocab_path)
    delay_spans = load_delay_spans(delay_path)
    validate_delay_spans(delay_spans)
    manifest = load_json_if_exists(manifest_path)

    discovered_splits = sorted(
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    )
    missing_expected = [name for name in EXPECTED_SPLITS if name not in discovered_splits]
    if missing_expected:
        fail(f"Missing expected split directories: {missing_expected}")

    split_results = {}
    total_sequences = 0
    for split_name in EXPECTED_SPLITS:
        split_dir = os.path.join(data_dir, split_name)
        info(f"Validating {split_name} ...")
        stats = validate_split(split_dir, split_name, dict_sys, dict_proc, delay_spans)
        split_results[split_name] = stats
        total_sequences += stats["n_sequences"]

    validate_manifest(
        manifest,
        split_results,
        dict_sys,
        dict_proc,
        delay_spans,
        args.strict_manifest,
    )

    info("")
    info("Dataset validation passed.")
    info(f"sys vocab size : {len(dict_sys)}")
    info(f"proc vocab size: {len(dict_proc)}")
    info(f"delay events   : {len(delay_spans)}")
    info(f"total sequences: {total_sequences}")

    if args.show_split_stats:
        info("")
        for split_name in EXPECTED_SPLITS:
            stats = split_results[split_name]
            info(
                f"{split_name}: shards={stats['n_shards']} seqs={stats['n_sequences']} "
                f"anom={stats['is_anomaly']} min_len={stats['min_seq_len']} "
                f"max_len={stats['max_seq_len']} trunc_rate={stats['truncation_rate']:.3%}"
            )


if __name__ == "__main__":
    main()
