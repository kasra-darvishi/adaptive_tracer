#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_DICT_PATH = os.path.join(os.path.dirname(_HERE), "dataset", "Dictionary.py")
_DICT_SPEC = importlib.util.spec_from_file_location("lmat_dictionary", _DICT_PATH)
if _DICT_SPEC is None or _DICT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load Dictionary.py from {_DICT_PATH}")
_DICT_MODULE = importlib.util.module_from_spec(_DICT_SPEC)
_DICT_SPEC.loader.exec_module(_DICT_MODULE)
Dictionary = _DICT_MODULE.Dictionary


class _DictionaryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "dataset.Dictionary" and name == "Dictionary":
            return Dictionary
        return super().find_class(module, name)


START_TOK = 2
END_TOK = 3
TRUNC_TOK = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate one LMAT NPZ shard plus vocab/delay files."
    )
    p.add_argument("--data_dir", required=True, help="Directory containing vocab.pkl and delay_spans.pkl")
    p.add_argument("--split", default="train_id", help="Split subdirectory containing shard_*.npz")
    p.add_argument("--shard", default="shard_000000.npz", help="Shard filename inside the split directory")
    p.add_argument("--expect_anomaly", type=int, choices=[0, 1], default=None)
    p.add_argument("--show_samples", type=int, default=2)
    return p.parse_args()


def fail(msg: str) -> None:
    raise AssertionError(msg)


def info(msg: str) -> None:
    print(msg, flush=True)


def load_vocab(path: str):
    with open(path, "rb") as fh:
        obj = _DictionaryUnpickler(fh).load()
    if not isinstance(obj, tuple) or len(obj) != 2:
        fail(f"Unexpected vocab.pkl structure in {path}")
    dict_sys, dict_proc = obj
    if not isinstance(dict_sys, Dictionary) or not isinstance(dict_proc, Dictionary):
        fail("vocab.pkl does not contain (Dictionary, Dictionary)")
    return dict_sys, dict_proc


def load_delay_spans(path: str):
    with open(path, "rb") as fh:
        delay_spans = pickle.load(fh)
    if not isinstance(delay_spans, dict):
        fail("delay_spans.pkl is not a dictionary")
    return delay_spans


def validate_delay_spans(delay_spans: dict) -> None:
    for event_name, value in delay_spans.items():
        if not isinstance(event_name, str):
            fail("delay_spans contains a non-string key")
        if not isinstance(value, tuple) or len(value) != 2:
            fail(f"delay_spans[{event_name!r}] is not a (boundaries, count) tuple")
        boundaries, count = value
        arr = np.asarray(boundaries, dtype=np.float64)
        if arr.ndim != 1:
            fail(f"delay_spans[{event_name!r}] boundaries must be 1D")
        if arr.size and not np.all(arr[:-1] <= arr[1:]):
            fail(f"delay_spans[{event_name!r}] boundaries are not sorted")
        if int(count) < 1:
            fail(f"delay_spans[{event_name!r}] count must be positive")


def validate_shard(npz_path: str, dict_sys: Dictionary, dict_proc: Dictionary, delay_spans: dict, expect_anomaly: int | None):
    required = {
        "call": np.int32,
        "entry": np.int8,
        "duration": np.int64,
        "proc": np.int32,
        "pid": np.int32,
        "tid": np.int32,
        "ret": np.int8,
        "lat_cat": np.uint8,
        "seq_len": np.int32,
        "req_dur_ms": np.float32,
        "is_anomaly": np.int8,
    }
    with np.load(npz_path, allow_pickle=False) as shard:
        keys = set(shard.files)
        missing = sorted(set(required) - keys)
        extra = sorted(keys - set(required))
        if missing:
            fail(f"Missing shard keys: {missing}")
        if extra:
            info(f"Note: extra shard keys present: {extra}")

        arrays = {k: shard[k] for k in required}

    n_seq = int(arrays["seq_len"].shape[0])
    if n_seq == 0:
        fail("Shard contains zero sequences")

    call = arrays["call"]
    if call.ndim != 2:
        fail("call must be a 2D array")
    n_rows, max_len = call.shape
    if n_rows != n_seq:
        fail("seq_len row count does not match call rows")

    for key, expected_dtype in required.items():
        arr = arrays[key]
        if arr.dtype != np.dtype(expected_dtype):
            fail(f"{key} dtype {arr.dtype} != expected {np.dtype(expected_dtype)}")

    two_d = ["call", "entry", "duration", "proc", "pid", "tid", "ret", "lat_cat"]
    one_d = ["seq_len", "req_dur_ms", "is_anomaly"]
    for key in two_d:
        if arrays[key].shape != (n_seq, max_len):
            fail(f"{key} shape {arrays[key].shape} != {(n_seq, max_len)}")
    for key in one_d:
        if arrays[key].shape != (n_seq,):
            fail(f"{key} shape {arrays[key].shape} != {(n_seq,)}")

    seq_len = arrays["seq_len"]
    if np.any(seq_len < 2):
        fail("Found seq_len < 2")
    if np.any(seq_len > max_len):
        fail("Found seq_len larger than padded shard width")

    if np.any(call[:, 0] != START_TOK):
        fail("Not all sequences start with START token in call[:, 0]")
    if np.any(arrays["proc"][:, 0] != dict_proc.get_idx("[START]")):
        fail("Not all sequences start with [START] proc token")
    if np.any(arrays["entry"][:, 0] != 0):
        fail("entry[:, 0] should be 0 for the START token")

    last_tokens = call[np.arange(n_seq), seq_len - 1]
    if np.any((last_tokens != END_TOK) & (last_tokens != TRUNC_TOK)):
        fail("Final token is neither END nor TRUNCATE for some sequences")

    if np.any(call < 0) or np.max(call) >= len(dict_sys):
        fail("call contains out-of-range vocabulary ids")
    if np.any(arrays["proc"] < 0) or np.max(arrays["proc"]) >= len(dict_proc):
        fail("proc contains out-of-range vocabulary ids")

    if not set(np.unique(arrays["entry"]).tolist()).issubset({0, 1, 2}):
        fail("entry contains values outside {0,1,2}")
    if not set(np.unique(arrays["ret"]).tolist()).issubset({0, 1, 2}):
        fail("ret contains values outside {0,1,2}")

    if np.any(arrays["duration"] < 0):
        fail("duration contains negative values")
    if np.any(arrays["pid"] < 0):
        fail("pid contains negative values")
    if np.any(arrays["tid"] < 0):
        fail("tid contains negative values")
    if np.any(arrays["req_dur_ms"] < 0):
        fail("req_dur_ms contains negative values")

    if expect_anomaly is not None and np.any(arrays["is_anomaly"] != expect_anomaly):
        fail(f"is_anomaly contains values other than expected {expect_anomaly}")

    observed_lat = int(np.max(arrays["lat_cat"]))
    theoretical_max = 0
    if delay_spans:
        theoretical_max = max(len(np.asarray(bounds)) + 1 for bounds, _ in delay_spans.values())
    if theoretical_max and observed_lat > theoretical_max:
        fail(f"lat_cat max {observed_lat} exceeds expected maximum {theoretical_max}")

    return arrays


def print_samples(arrays, dict_sys: Dictionary, dict_proc: Dictionary, n_show: int) -> None:
    n_seq = arrays["seq_len"].shape[0]
    for idx in range(min(n_show, n_seq)):
        sl = int(arrays["seq_len"][idx])
        call_ids = arrays["call"][idx, :sl].tolist()
        proc_ids = arrays["proc"][idx, :sl].tolist()
        call_names = [dict_sys.idx2word[token] for token in call_ids[:12]]
        proc_names = [dict_proc.idx2word[token] for token in proc_ids[:8]]
        info(
            f"sample[{idx}] seq_len={sl} req_dur_ms={float(arrays['req_dur_ms'][idx]):.3f} "
            f"is_anomaly={int(arrays['is_anomaly'][idx])}"
        )
        info(f"  call[:12] -> {call_names}")
        info(f"  proc[:8]  -> {proc_names}")
        info(f"  entry[:12]= {arrays['entry'][idx, :min(sl, 12)].tolist()}")
        info(f"  ret[:12]  = {arrays['ret'][idx, :min(sl, 12)].tolist()}")
        info(f"  lat[:12]  = {arrays['lat_cat'][idx, :min(sl, 12)].tolist()}")


def main() -> None:
    args = parse_args()
    vocab_path = os.path.join(args.data_dir, "vocab.pkl")
    delay_path = os.path.join(args.data_dir, "delay_spans.pkl")
    shard_path = os.path.join(args.data_dir, args.split, args.shard)

    if not os.path.isfile(vocab_path):
        fail(f"Missing file: {vocab_path}")
    if not os.path.isfile(delay_path):
        fail(f"Missing file: {delay_path}")
    if not os.path.isfile(shard_path):
        fail(f"Missing file: {shard_path}")

    dict_sys, dict_proc = load_vocab(vocab_path)
    delay_spans = load_delay_spans(delay_path)
    validate_delay_spans(delay_spans)
    arrays = validate_shard(shard_path, dict_sys, dict_proc, delay_spans, args.expect_anomaly)

    info("Validation passed.")
    info(f"sys vocab size : {len(dict_sys)}")
    info(f"proc vocab size: {len(dict_proc)}")
    info(f"delay events   : {len(delay_spans)}")
    info(f"n_sequences    : {int(arrays['seq_len'].shape[0])}")
    info(f"shard_width    : {int(arrays['call'].shape[1])}")
    info(f"lat_cat max    : {int(np.max(arrays['lat_cat']))}")
    print_samples(arrays, dict_sys, dict_proc, args.show_samples)


if __name__ == "__main__":
    main()
