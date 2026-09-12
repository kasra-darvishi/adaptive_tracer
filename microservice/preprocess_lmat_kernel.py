#!/usr/bin/env python3
"""
preprocess_lmat_kernel.py
=========================

Clean LMAT preprocessing pipeline for SockShop kernel traces.

This script builds LMAT training shards directly from raw LTTng kernel traces
using the methodology described in the paper:

1. Model system behavior as timestamp-ordered event sequences.
2. Represent each event with syscall name, process name, pid, tid, delay,
   entry/exit type, and return-status signal.
3. Recover execution durations by matching entry/exit pairs per event type and
   execution context (thread id).
4. Discretize durations into ordinal categories using training-set latency
   boundaries.
5. Write padded NPZ shards that are consumed by train_sockshop.py.

This dataset supports all three paper training regimes without reprocessing:
  - Event-only
  - Duration-only
  - Multi-task

Only the training flags change. The shard format remains the same.

Default split layout for five runs per scenario:
  - train_id: normal/run01, normal/run02, normal/run03
  - valid_id: normal/run04
  - test_id:  normal/run05
  - valid/test OOD: run04/run05 for each anomaly family

Expected output layout:
  <output_dir>/
    vocab.pkl
    delay_spans.pkl
    dataset_manifest.json
    train_id/
    valid_id/
    test_id/
    valid_ood_cpu/
    test_ood_cpu/
    valid_ood_disk/
    test_ood_disk/
    valid_ood_mem/
    test_ood_mem/
    valid_ood_net/
    test_ood_net/
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
from array import array
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import time

import numpy as np

try:
    import bt2

    BT2_AVAILABLE = True
except ImportError:
    BT2_AVAILABLE = False


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from dataset.Dictionary import Dictionary


START_TOK = 2
END_TOK = 3
TRUNC_TOK = 4

JOB_START = time()
FIELD_RE = re.compile(r'(\w+) = (?:"([^"]*)"|(-?\d+))')


def log(msg: str, prefix: str = "INFO") -> None:
    elapsed = timedelta(seconds=int(time() - JOB_START))
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now} +{elapsed}] [{prefix}] {msg}", flush=True)


@dataclass
class SplitSpec:
    name: str
    dirs: list[str]
    label: int


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build LMAT NPZ shards from raw SockShop kernel traces."
    )
    p.add_argument("--trace_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--txt_dump_dir", default=None)

    p.add_argument("--window_ms", type=float, default=100.0)
    p.add_argument("--warmup_s", type=float, default=5.0)
    p.add_argument("--min_events", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument(
        "--n_categories",
        type=int,
        default=6,
        choices=[4, 6, 8, 10],
        help="Total latency classes used in the shard labels, including the "
             "reserved 0 class for pad/non-exit tokens. Therefore 6 here is "
             "logically equivalent to the paper's 5 duration categories.",
    )
    p.add_argument(
        "--paper_duration_bins",
        type=int,
        choices=[3, 5, 7, 9],
        default=None,
        help="Paper-style duration-bin count excluding the reserved 0 class. "
             "If set, the script maps 3/5/7/9 to effective n_categories "
             "4/6/8/10 respectively.",
    )
    p.add_argument("--shard_size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--event_scope",
        choices=["syscall", "all"],
        default="syscall",
        help="LMAT paper-aligned mode is 'syscall'. Use 'all' to include "
             "other kernel tracepoints such as sched/timer/rcu events.",
    )

    p.add_argument("--normal_dir", default="normal")
    p.add_argument("--cpu_dir", default="anomaly_cpu")
    p.add_argument("--disk_dir", default="anomaly_disk")
    p.add_argument("--mem_dir", default="anomaly_mem")
    p.add_argument("--net_dir", default="anomaly_net")

    p.add_argument("--normal_train_runs", default="run01,run02,run03")
    p.add_argument("--normal_valid_runs", default="run04")
    p.add_argument("--normal_test_runs", default="run05")
    p.add_argument("--ood_valid_runs", default="run04")
    p.add_argument("--ood_test_runs", default="run05")

    return p.parse_args()


def build_split_specs(args: argparse.Namespace) -> list[SplitSpec]:
    def join_runs(scenario_dir: str, runs: list[str]) -> list[str]:
        return [f"{scenario_dir}/{run}" for run in runs]

    valid_ood = csv_list(args.ood_valid_runs)
    test_ood = csv_list(args.ood_test_runs)

    return [
        SplitSpec("train_id", join_runs(args.normal_dir, csv_list(args.normal_train_runs)), 0),
        SplitSpec("valid_id", join_runs(args.normal_dir, csv_list(args.normal_valid_runs)), 0),
        SplitSpec("test_id", join_runs(args.normal_dir, csv_list(args.normal_test_runs)), 0),
        SplitSpec("valid_ood_cpu", join_runs(args.cpu_dir, valid_ood), 1),
        SplitSpec("test_ood_cpu", join_runs(args.cpu_dir, test_ood), 1),
        SplitSpec("valid_ood_disk", join_runs(args.disk_dir, valid_ood), 1),
        SplitSpec("test_ood_disk", join_runs(args.disk_dir, test_ood), 1),
        SplitSpec("valid_ood_mem", join_runs(args.mem_dir, valid_ood), 1),
        SplitSpec("test_ood_mem", join_runs(args.mem_dir, test_ood), 1),
        SplitSpec("valid_ood_net", join_runs(args.net_dir, valid_ood), 1),
        SplitSpec("test_ood_net", join_runs(args.net_dir, test_ood), 1),
    ]


def resolve_run_dir(trace_root: str, run_rel: str) -> str:
    candidates = [
        os.path.join(trace_root, run_rel),
        os.path.join(trace_root, "traces", run_rel),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def find_text_dump(txt_dump_dir: str | None, run_dir: str) -> str | None:
    if not txt_dump_dir:
        return None
    parts = run_dir.replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2:
        return None
    scenario = parts[-2]
    run_name = parts[-1]
    candidates = [
        os.path.join(txt_dump_dir, scenario, run_name, "kernel.txt"),
        os.path.join(txt_dump_dir, f"{scenario}_{run_name}_kernel.txt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def find_ctf_root(base_dir: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        if any(name.endswith(".idx") for name in files):
            return root
        if "metadata" in files:
            return root
    return base_dir


def safe_field(ev, names, default):
    for name in names:
        try:
            return ev[name]
        except (KeyError, TypeError):
            continue
    return default


def parse_kernel_text_dump(txt_path: str, warmup_s: float):
    log(f"Parsing text dump: {txt_path}", prefix="PARSE")
    first_abs_ts = None
    warmup_ns = int(warmup_s * 1e9)
    prev_sec = -1
    day_offset = 0
    count = 0

    with open(txt_path, "r", errors="replace") as fh:
        for line in fh:
            count += 1
            ts_start = line.find("[")
            ts_end = line.find("]", ts_start)
            if ts_start < 0 or ts_end < 0:
                continue

            ts_str = line[ts_start + 1 : ts_end]
            dot = ts_str.rfind(".")
            if dot < 0:
                continue

            try:
                h, m, s = ts_str[:dot].split(":")
                cur_sec = int(h) * 3600 + int(m) * 60 + int(s)
                if prev_sec >= 0 and cur_sec < prev_sec - 3600:
                    day_offset += 86400
                prev_sec = cur_sec
                ns = (cur_sec + day_offset) * 1_000_000_000 + int(ts_str[dot + 1 :].ljust(9, "0"))
            except (ValueError, IndexError):
                continue

            if first_abs_ts is None:
                first_abs_ts = ns
            if ns - first_abs_ts < warmup_ns:
                continue

            after_delta = line.find(")", ts_end)
            if after_delta < 0:
                continue
            rest = line[after_delta + 2 :]
            sp1 = rest.find(" ")
            sp2 = rest.find(" ", sp1 + 1)
            if sp1 < 0 or sp2 < 0:
                continue

            raw_name = rest[sp1 + 1 : sp2].rstrip(":")
            name = raw_name.replace("kernel:", "")
            name = name.replace("syscall_", "").replace("entry_", "").replace("exit_", "")
            is_entry = "entry" in raw_name
            is_exit = "exit" in raw_name

            pid = 0
            tid = 0
            procname = "unknown"
            ret_enc = 0
            for match in FIELD_RE.finditer(rest[sp2:]):
                field_name = match.group(1)
                sval = match.group(2)
                ival = match.group(3)
                if field_name == "pid" and ival:
                    pid = int(ival)
                elif field_name == "tid" and ival:
                    tid = int(ival)
                elif field_name == "procname" and sval:
                    procname = sval
                elif field_name == "ret" and ival:
                    ret_val = int(ival)
                    ret_enc = 1 if ret_val >= 0 else 2

            yield {
                "name": name,
                "raw_name": raw_name,
                "timestamp": ns,
                "pid": pid,
                "tid": tid,
                "procname": procname,
                "ret": ret_enc,
                "is_entry": is_entry,
                "is_exit": is_exit,
            }

            if count % 5_000_000 == 0:
                log(f"Read {count:,} text lines", prefix="PARSE")


def parse_kernel_bt2(kernel_trace_dir: str, warmup_s: float):
    if not BT2_AVAILABLE:
        raise RuntimeError("bt2 is not available and no text dump was provided.")

    ctf_root = find_ctf_root(kernel_trace_dir)
    if ctf_root is None:
        raise FileNotFoundError(f"No kernel trace found under {kernel_trace_dir}")

    log(f"Parsing bt2 trace: {ctf_root}", prefix="PARSE")
    iterator = bt2.TraceCollectionMessageIterator(ctf_root)
    first_ts = None
    count = 0
    warmup_ns = int(warmup_s * 1e9)

    for msg in iterator:
        if not isinstance(msg, bt2._EventMessageConst):
            continue
        ev = msg.event
        ts = msg.default_clock_snapshot.ns_from_origin
        count += 1

        if first_ts is None:
            first_ts = ts
        if ts - first_ts < warmup_ns:
            continue

        raw_name = ev.name
        name = raw_name.replace("kernel:", "")
        name = name.replace("syscall_", "").replace("entry_", "").replace("exit_", "")
        is_entry = "entry" in raw_name
        is_exit = "exit" in raw_name

        pid = int(safe_field(ev, ("vpid", "pid"), 0))
        tid = int(safe_field(ev, ("vtid", "tid"), 0))
        procname = str(safe_field(ev, ("procname",), "unknown"))
        ret_raw = safe_field(ev, ("ret",), None)
        if ret_raw is None:
            ret_enc = 0
        else:
            ret_enc = 1 if int(ret_raw) >= 0 else 2

        yield {
            "name": name,
            "raw_name": raw_name,
            "timestamp": ts,
            "pid": pid,
            "tid": tid,
            "procname": procname,
            "ret": ret_enc,
            "is_entry": is_entry,
            "is_exit": is_exit,
        }

        if count % 500_000 == 0:
            log(f"Read {count:,} bt2 events", prefix="PARSE")


def stream_kernel_events(run_dir: str, txt_dump_dir: str | None, warmup_s: float):
    text_dump = find_text_dump(txt_dump_dir, run_dir)
    if text_dump:
        yield from parse_kernel_text_dump(text_dump, warmup_s)
        return

    kernel_dir = os.path.join(run_dir, "kernel")
    yield from parse_kernel_bt2(kernel_dir, warmup_s)


def is_syscall_event(event: dict) -> bool:
    raw_name = event.get("raw_name", "")
    return "syscall_" in raw_name


def filter_events_by_scope(event_iter, event_scope: str):
    stats = {
        "events_total": 0,
        "events_kept": 0,
        "events_dropped": 0,
        "syscall_events": 0,
        "nonsyscall_events": 0,
    }

    for event in event_iter:
        stats["events_total"] += 1
        syscall = is_syscall_event(event)
        if syscall:
            stats["syscall_events"] += 1
        else:
            stats["nonsyscall_events"] += 1

        if event_scope == "all" or syscall:
            stats["events_kept"] += 1
            yield event, stats
        else:
            stats["events_dropped"] += 1


def segment_time_windows_with_stats(event_iter, window_ms: float, min_events: int):
    """Segment filtered events while preserving running event stats."""
    window_ns = int(window_ms * 1e6)
    buffers = defaultdict(list)
    starts = {}
    latest_stats = None
    windows_emitted = 0

    for ev, stats in event_iter:
        latest_stats = dict(stats)
        tid = ev["tid"]
        ts = ev["timestamp"]
        if tid not in starts:
            starts[tid] = ts

        elapsed = ts - starts[tid]
        if elapsed >= window_ns and buffers[tid]:
            buf = buffers[tid]
            dur_ms = elapsed / 1e6
            if len(buf) >= min_events:
                windows_emitted += 1
                yield list(buf), dur_ms, None
            buffers[tid] = [ev]
            starts[tid] = ts
        else:
            buffers[tid].append(ev)

        if stats["events_total"] % 5_000_000 == 0:
            log(
                f"Scope={stats['events_kept']:,}/{stats['events_total']:,} kept, "
                f"{windows_emitted:,} windows",
                prefix="SEG",
            )

    for tid, buf in buffers.items():
        if len(buf) >= min_events and tid in starts:
            dur_ms = (buf[-1]["timestamp"] - starts[tid]) / 1e6
            windows_emitted += 1
            yield list(buf), dur_ms, None

    if latest_stats is None:
        latest_stats = {
            "events_total": 0,
            "events_kept": 0,
            "events_dropped": 0,
            "syscall_events": 0,
            "nonsyscall_events": 0,
        }

    latest_stats["windows_emitted"] = windows_emitted
    yield None, None, latest_stats


def segment_time_windows(event_iter, window_ms: float, min_events: int):
    window_ns = int(window_ms * 1e6)
    buffers = defaultdict(list)
    starts = {}
    sequences = []
    event_count = 0

    for ev in event_iter:
        event_count += 1
        tid = ev["tid"]
        ts = ev["timestamp"]
        if tid not in starts:
            starts[tid] = ts

        elapsed = ts - starts[tid]
        if elapsed >= window_ns and buffers[tid]:
            buf = buffers[tid]
            dur_ms = elapsed / 1e6
            if len(buf) >= min_events:
                sequences.append((list(buf), dur_ms))
            buffers[tid] = [ev]
            starts[tid] = ts
        else:
            buffers[tid].append(ev)

        if event_count % 5_000_000 == 0:
            log(f"Segmented {event_count:,} events into {len(sequences):,} windows", prefix="SEG")

    for tid, buf in buffers.items():
        if len(buf) >= min_events and tid in starts:
            dur_ms = (buf[-1]["timestamp"] - starts[tid]) / 1e6
            sequences.append((list(buf), dur_ms))

    return sequences


def encode_sequence(events, dict_sys: Dictionary, dict_proc: Dictionary, is_train: bool, max_seq_len: int):
    call = [START_TOK]
    entry = [0]
    duration = [0]
    proc = [dict_proc.get_idx("[START]")]
    pid = [0]
    tid = [0]
    ret = [0]
    raw_lat = [0]
    ev_names = [""]

    entry_times = {}
    prev_ts = None

    for ev in events:
        sysname = ev["name"]
        procname = ev["procname"]
        ts = ev["timestamp"]

        if is_train:
            dict_sys.add_word(sysname)
            dict_proc.add_word(procname)

        call.append(dict_sys.get_idx(sysname))
        proc.append(dict_proc.get_idx(procname))

        entry_val = 1 if ev["is_entry"] else (2 if ev["is_exit"] else 0)
        entry.append(entry_val)

        delta = 0 if prev_ts is None else max(0, ts - prev_ts)
        duration.append(delta)
        prev_ts = ts

        pid.append(ev["pid"])
        tid.append(ev["tid"])
        ret.append(ev["ret"])

        key = (sysname, ev["tid"])
        latency = 0
        if ev["is_entry"]:
            entry_times[key] = ts
        elif ev["is_exit"] and key in entry_times:
            latency = max(0, ts - entry_times.pop(key))

        raw_lat.append(latency)
        ev_names.append(ev["raw_name"])

    call.append(END_TOK)
    proc.append(dict_proc.get_idx("[END]"))
    entry.append(0)
    duration.append(0)
    pid.append(0)
    tid.append(0)
    ret.append(0)
    raw_lat.append(0)
    ev_names.append("")

    if len(call) > max_seq_len:
        call = call[: max_seq_len - 1] + [TRUNC_TOK]
        proc = proc[: max_seq_len - 1] + [TRUNC_TOK]
        entry = entry[:max_seq_len]
        duration = duration[:max_seq_len]
        pid = pid[:max_seq_len]
        tid = tid[:max_seq_len]
        ret = ret[:max_seq_len]
        raw_lat = raw_lat[:max_seq_len]
        ev_names = ev_names[:max_seq_len]

    return {
        "call": call,
        "entry": entry,
        "duration": duration,
        "proc": proc,
        "pid": pid,
        "tid": tid,
        "ret": ret,
        "raw_lat": raw_lat,
        "ev_names": ev_names,
        "seq_len": len(call),
    }


def build_delay_spans(all_latencies, all_names, n_categories: int):
    n_cat = n_categories - 1
    total = n_cat * (n_cat + 1) / 2
    event_latencies = defaultdict(list)

    for latencies, names in zip(all_latencies, all_names):
        for latency, raw_name in zip(latencies, names):
            if "exit" not in raw_name or latency <= 0:
                continue
            base = raw_name.replace("syscall_", "").replace("exit_", "")
            event_latencies[base].append(latency)

    delay_spans = {}
    for event_name, values in event_latencies.items():
        if len(values) < 2:
            continue
        arr = np.sort(np.asarray(values, dtype=np.float64))
        percentiles = []
        cumulative = 0.0
        for idx in range(1, n_cat):
            fraction = (n_cat - idx + 1) / total
            cumulative += fraction * 100
            percentiles.append(cumulative)
        boundaries = np.percentile(arr, percentiles)
        delay_spans[event_name] = (boundaries, len(values))
    return delay_spans


def categorise_latency(raw_lat, ev_names, delay_spans):
    cats = np.zeros(len(raw_lat), dtype=np.uint8)
    for idx, (latency, raw_name) in enumerate(zip(raw_lat, ev_names)):
        if "exit" not in raw_name or latency == 0:
            continue
        base = raw_name.replace("syscall_", "").replace("exit_", "")
        info = delay_spans.get(base)
        if info is None:
            continue
        boundaries, _ = info
        cat = int(np.searchsorted(boundaries, latency, side="left")) + 1
        cats[idx] = min(cat, len(boundaries) + 1)
    return cats


def apply_latency_categories(sequences, delay_spans):
    for seq in sequences:
        seq["lat_cat"] = categorise_latency(seq["raw_lat"], seq["ev_names"], delay_spans)


def iter_exit_latencies(sequence):
    for latency, raw_name in zip(sequence["raw_lat"], sequence["ev_names"]):
        if "exit" not in raw_name or latency <= 0:
            continue
        base = raw_name.replace("syscall_", "").replace("exit_", "")
        yield base, int(latency)


def append_latency_bins(latency_dir: str, sequences) -> None:
    os.makedirs(latency_dir, exist_ok=True)
    grouped = defaultdict(list)
    for seq in sequences:
        for base, latency in iter_exit_latencies(seq):
            grouped[base].append(latency)

    for base, values in grouped.items():
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
        path = os.path.join(latency_dir, f"{safe_name}.bin")
        with open(path, "ab") as fh:
            array("Q", values).tofile(fh)


def build_delay_spans_from_latency_dir(latency_dir: str, n_categories: int):
    n_cat = n_categories - 1
    total = n_cat * (n_cat + 1) / 2
    delay_spans = {}

    if not os.path.isdir(latency_dir):
        return delay_spans

    for name in sorted(os.listdir(latency_dir)):
        if not name.endswith(".bin"):
            continue
        path = os.path.join(latency_dir, name)
        values = np.fromfile(path, dtype=np.uint64)
        if values.size < 2:
            continue
        arr = np.sort(values.astype(np.float64))
        percentiles = []
        cumulative = 0.0
        for idx in range(1, n_cat):
            fraction = (n_cat - idx + 1) / total
            cumulative += fraction * 100
            percentiles.append(cumulative)
        boundaries = np.percentile(arr, percentiles)
        event_name = name[:-4]
        delay_spans[event_name] = (boundaries, int(values.size))
    return delay_spans


def pad_array(batch, key: str, max_len: int, pad_value: int, dtype):
    out = np.full((len(batch), max_len), pad_value, dtype=dtype)
    for idx, seq in enumerate(batch):
        values = seq[key]
        out[idx, : len(values)] = values
    return out


def write_shards(sequences, split_dir: str, shard_size: int, is_anomaly: int, meta_extra: dict):
    os.makedirs(split_dir, exist_ok=True)
    n_total = len(sequences)
    n_shards = (n_total + shard_size - 1) // shard_size if n_total else 0
    log(f"Writing {n_total:,} sequences to {split_dir}", prefix="SHARD")

    written = 0
    for shard_idx in range(n_shards):
        batch = sequences[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        if not batch:
            break
        max_len = max(seq["seq_len"] for seq in batch)
        shard = {
            "call": pad_array(batch, "call", max_len, 0, np.int32),
            "entry": pad_array(batch, "entry", max_len, 0, np.int8),
            "duration": pad_array(batch, "duration", max_len, 0, np.int64),
            "proc": pad_array(batch, "proc", max_len, 0, np.int32),
            "pid": pad_array(batch, "pid", max_len, 0, np.int32),
            "tid": pad_array(batch, "tid", max_len, 0, np.int32),
            "ret": pad_array(batch, "ret", max_len, 0, np.int8),
            "lat_cat": pad_array(batch, "lat_cat", max_len, 0, np.uint8),
            "seq_len": np.asarray([seq["seq_len"] for seq in batch], dtype=np.int32),
            "req_dur_ms": np.asarray([seq["req_dur_ms"] for seq in batch], dtype=np.float32),
            "is_anomaly": np.full(len(batch), is_anomaly, dtype=np.int8),
        }
        np.savez_compressed(os.path.join(split_dir, f"shard_{shard_idx:06d}.npz"), **shard)
        written += len(batch)

    meta = {
        "n_sequences": written,
        "n_shards": n_shards,
        "is_anomaly": is_anomaly,
        "supported_training_modes": ["event", "duration", "multitask"],
    }
    meta.update(meta_extra)
    with open(os.path.join(split_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


class StreamingShardWriter:
    def __init__(self, split_dir: str, shard_size: int, is_anomaly: int, meta_extra: dict):
        self.split_dir = split_dir
        self.shard_size = shard_size
        self.is_anomaly = is_anomaly
        self.meta_extra = meta_extra
        self.buffer = []
        self.written = 0
        self.shard_idx = 0
        os.makedirs(split_dir, exist_ok=True)

    def _flush_batch(self, batch):
        if not batch:
            return
        max_len = max(seq["seq_len"] for seq in batch)
        shard = {
            "call": pad_array(batch, "call", max_len, 0, np.int32),
            "entry": pad_array(batch, "entry", max_len, 0, np.int8),
            "duration": pad_array(batch, "duration", max_len, 0, np.int64),
            "proc": pad_array(batch, "proc", max_len, 0, np.int32),
            "pid": pad_array(batch, "pid", max_len, 0, np.int32),
            "tid": pad_array(batch, "tid", max_len, 0, np.int32),
            "ret": pad_array(batch, "ret", max_len, 0, np.int8),
            "lat_cat": pad_array(batch, "lat_cat", max_len, 0, np.uint8),
            "seq_len": np.asarray([seq["seq_len"] for seq in batch], dtype=np.int32),
            "req_dur_ms": np.asarray([seq["req_dur_ms"] for seq in batch], dtype=np.float32),
            "is_anomaly": np.full(len(batch), self.is_anomaly, dtype=np.int8),
        }
        np.savez_compressed(
            os.path.join(self.split_dir, f"shard_{self.shard_idx:06d}.npz"),
            **shard,
        )
        self.shard_idx += 1
        self.written += len(batch)

    def add_sequences(self, sequences):
        for seq in sequences:
            self.buffer.append(seq)
            if len(self.buffer) >= self.shard_size:
                batch = self.buffer[: self.shard_size]
                self.buffer = self.buffer[self.shard_size :]
                self._flush_batch(batch)

    def close(self):
        if self.buffer:
            self._flush_batch(self.buffer)
            self.buffer = []

        meta = {
            "n_sequences": self.written,
            "n_shards": self.shard_idx,
            "is_anomaly": self.is_anomaly,
            "supported_training_modes": ["event", "duration", "multitask"],
        }
        meta.update(self.meta_extra)
        with open(os.path.join(self.split_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)


def process_run_streaming(
    run_dir: str,
    dict_sys: Dictionary,
    dict_proc: Dictionary,
    args,
    is_train: bool,
    on_batch,
):
    log(f"Run: {run_dir}", prefix="RUN")
    raw_events = stream_kernel_events(run_dir, args.txt_dump_dir, args.warmup_s)
    filtered_events = filter_events_by_scope(raw_events, args.event_scope)
    encoded_batch = []
    run_stats = None

    for window_events, dur_ms, final_stats in segment_time_windows_with_stats(
        filtered_events, args.window_ms, args.min_events
    ):
        if final_stats is not None:
            run_stats = final_stats
            break

        seq = encode_sequence(window_events, dict_sys, dict_proc, is_train, args.max_seq_len)
        seq["req_dur_ms"] = dur_ms
        encoded_batch.append(seq)

        if len(encoded_batch) >= args.shard_size:
            on_batch(encoded_batch)
            encoded_batch = []

    if encoded_batch:
        on_batch(encoded_batch)

    if run_stats is None:
        run_stats = {
            "events_total": 0,
            "events_kept": 0,
            "events_dropped": 0,
            "syscall_events": 0,
            "nonsyscall_events": 0,
            "windows_emitted": 0,
        }

    log(
        f"Produced {run_stats['windows_emitted']:,} windows | kept "
        f"{run_stats['events_kept']:,}/{run_stats['events_total']:,} events "
        f"(syscall={run_stats['syscall_events']:,}, "
        f"other={run_stats['nonsyscall_events']:,})",
        prefix="RUN",
    )
    return run_stats


def build_train_artifacts(train_spec: SplitSpec, args):
    dict_sys = Dictionary()
    dict_proc = Dictionary()
    run_summaries = []
    latency_dir = os.path.join(args.output_dir, ".tmp_train_latency")

    if os.path.isdir(latency_dir):
        shutil.rmtree(latency_dir)
    os.makedirs(latency_dir, exist_ok=True)

    for run_rel in train_spec.dirs:
        run_dir = resolve_run_dir(args.trace_root, run_rel)
        if not os.path.isdir(run_dir):
            log(f"Missing run directory: {run_dir}", prefix="WARN")
            continue
        run_state = {"n_sequences": 0}

        def handle_batch(batch):
            append_latency_bins(latency_dir, batch)
            run_state["n_sequences"] += len(batch)

        run_stats = process_run_streaming(
            run_dir,
            dict_sys,
            dict_proc,
            args,
            is_train=True,
            on_batch=handle_batch,
        )
        run_summaries.append({"run": run_rel, **run_stats, "n_sequences": run_state["n_sequences"]})

    if not run_summaries:
        raise RuntimeError("Training split produced zero sequences.")

    delay_spans = build_delay_spans_from_latency_dir(latency_dir, args.n_categories)
    shutil.rmtree(latency_dir, ignore_errors=True)
    return dict_sys, dict_proc, delay_spans, run_summaries


def stream_split_to_shards(
    spec: SplitSpec,
    dict_sys: Dictionary,
    dict_proc: Dictionary,
    delay_spans,
    args,
    *,
    is_train_pass: bool,
):
    run_summaries = []
    writer = StreamingShardWriter(
        os.path.join(args.output_dir, spec.name),
        args.shard_size,
        spec.label,
        {
            "dirs": spec.dirs,
            "n_categories": args.n_categories,
            "event_scope": args.event_scope,
            "run_summaries": run_summaries,
        },
    )

    for run_rel in spec.dirs:
        run_dir = resolve_run_dir(args.trace_root, run_rel)
        if not os.path.isdir(run_dir):
            log(f"Missing run directory: {run_dir}", prefix="WARN")
            continue
        run_state = {"n_sequences": 0}

        def handle_batch(batch):
            apply_latency_categories(batch, delay_spans)
            writer.add_sequences(batch)
            run_state["n_sequences"] += len(batch)

        run_stats = process_run_streaming(
            run_dir,
            dict_sys,
            dict_proc,
            args,
            is_train=is_train_pass,
            on_batch=handle_batch,
        )
        run_summaries.append({"run": run_rel, **run_stats, "n_sequences": run_state["n_sequences"]})

    writer.close()
    return run_summaries


def main():
    args = parse_args()
    if args.paper_duration_bins is not None:
        args.n_categories = args.paper_duration_bins + 1
        log(
            f"paper_duration_bins={args.paper_duration_bins} -> "
            f"effective n_categories={args.n_categories}",
            prefix="START",
        )
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    split_specs = build_split_specs(args)
    manifest = {
        "methodology": {
            "window_ms": args.window_ms,
            "warmup_s": args.warmup_s,
            "min_events": args.min_events,
            "max_seq_len": args.max_seq_len,
            "n_categories": args.n_categories,
            "paper_duration_bins": (
                args.paper_duration_bins
                if args.paper_duration_bins is not None
                else args.n_categories - 1
            ),
            "event_scope": args.event_scope,
            "notes": [
                "Event representation follows LMAT syscall sequence modeling.",
                "Duration categories are built from training exits only.",
                "Code n_categories includes the reserved 0 class; the paper's "
                "3/5/7/9 notation refers to the non-zero duration bins only.",
                "The same shards support event-only, duration-only, and multi-task training.",
            ],
        },
        "splits": [{"name": spec.name, "dirs": spec.dirs, "label": spec.label} for spec in split_specs],
    }

    log("Building training split and vocabularies", prefix="START")
    train_spec = split_specs[0]
    dict_sys, dict_proc, delay_spans, train_run_summaries = build_train_artifacts(train_spec, args)

    with open(os.path.join(args.output_dir, "vocab.pkl"), "wb") as fh:
        pickle.dump((dict_sys, dict_proc), fh)
    with open(os.path.join(args.output_dir, "delay_spans.pkl"), "wb") as fh:
        pickle.dump(delay_spans, fh)

    manifest["vocab"] = {
        "n_syscall": len(dict_sys),
        "n_process": len(dict_proc),
        "n_delay_events": len(delay_spans),
    }
    manifest["split_summaries"] = {
        train_spec.name: train_run_summaries,
    }

    log(f"Writing split: {train_spec.name}", prefix="START")
    manifest["split_summaries"][train_spec.name] = stream_split_to_shards(
        train_spec,
        dict_sys,
        dict_proc,
        delay_spans,
        args,
        is_train_pass=False,
    )

    for spec in split_specs[1:]:
        log(f"Building split: {spec.name}", prefix="START")
        manifest["split_summaries"][spec.name] = stream_split_to_shards(
            spec,
            dict_sys,
            dict_proc,
            delay_spans,
            args,
            is_train_pass=False,
        )

    with open(os.path.join(args.output_dir, "dataset_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    log(f"Done. Output written to {args.output_dir}", prefix="DONE")


if __name__ == "__main__":
    main()
