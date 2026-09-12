#!/usr/bin/env python3
"""
preprocess_sockshop.py
======================

Converts raw LTTng CTF traces from the SockShop microservices demo into
ML-ready NumPy compressed (.npz) shards that can be loaded *much* faster
than the original semicolon-text format used by the Apache LMAT pipeline.

Two segmentation strategies are supported (selectable via --seg_mode):

  ust   (default) — use OTel span boundaries from the LTTng UST (user-space)
                    trace.  Each recorded span [span_start, span_end] defines
                    a time window; all kernel events on the same host that fall
                    inside that window are collected as one sequence.
                    This is the richest option but requires that both the
                    kernel AND the ust sub-directories exist.

  time  (fallback) — fixed-duration sliding windows per TID.  Used when the
                    UST trace is absent or too sparse.

Output
------
Each split produces a folder of NPZ shards:

    <output_dir>/<split_name>/
        shard_000000.npz   ...   shard_NNNNNN.npz
        vocab.pkl                dict_sys + dict_proc
        delay_spans.pkl          latency boundaries (train only, then frozen)
        meta.json                split statistics

Every shard holds up to --shard_size sequences (default 5000).  Each shard
NPZ contains parallel arrays (sequences are zero-padded to the longest seq in
the shard):

    call        int32  (L,)   system-call vocabulary index
    entry       int8   (L,)   0=none 1=entry 2=exit
    duration    int64  (L,)   ns since previous event
    proc        int32  (L,)   process-name vocabulary index
    pid         int32  (L,)   raw PID
    tid         int32  (L,)   raw TID
    ret         int8   (L,)   0=no-ret 1=success 2=failure
    lat_cat     uint8  (L,)   latency category (0=pad/entry, 1..n_cat)
    seq_len     int32  ()     actual unpadded length (for masking)
    req_dur_ms  float32()     wall-clock span/window duration in ms
    is_anomaly  int8   ()     0=normal 1=anomaly (set by --label flag)

The categorical latency in 'lat_cat' uses training-set percentile boundaries
(same algorithm as the original LMAT categorize_latency, but stored once per
shard rather than embedded inline in every text line).

Usage
-----
python microservice/preprocess_sockshop.py \\
    --trace_root  micro-service-trace-data/traces \\
    --output_dir  micro-service-trace-data/preprocessed \\
    --splits      "train_id:normal/run01,normal/run02,normal/run03:0" \\
                  "valid_id:normal/run04:0" \\
                  "test_id:normal/run05:0" \\
                  "valid_ood_cpu:cpu_stress/run01:1" \\
                  "test_ood_cpu:cpu_stress/run02:1" \\
                  "valid_ood_disk:disk_stress/run01:1" \\
                  "test_ood_disk:disk_stress/run02:1" \\
                  "valid_ood_mem:mem_stress/run01:1" \\
                  "test_ood_mem:mem_stress/run02:1" \\
    --seg_mode ust \\
    --window_ms 100 \\
    --warmup_s 5 \\
    --min_events 8 \\
    --n_categories 6 \\
    --shard_size 5000

To match paper-style experiments with different latency discretizations (e.g. 3, 5, 7, or 9
duration categories), rerun preprocessing with the desired ``--n_categories`` and train with
the same value. For Apache-comparable Event / Duration / Multi-task comparisons, see
``DOCS/dataset-microservice-explaination.md`` (``train_sockshop.py --ood_score``).
Those three training modes reuse the SAME preprocessed shards; only the training flags change.

Each --splits entry has the format:  split_name:run_dir1,run_dir2,...:label
  split_name  — output folder name (e.g. train_id, valid_ood_cpu)
  run_dirs    — comma-separated paths *relative to* --trace_root
  label       — 0=normal, 1=anomaly (written into is_anomaly field)

Alternatively, use ``--split_preset five_run_ood`` to auto-generate the
canonical 5-run SockShop layout:
  normal train = run01,run02,run03
  normal valid = run04
  normal test  = run05
  anomaly valid/test defaults = run04/run05 for each anomaly family
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from time import time
from datetime import datetime, timedelta
from collections import defaultdict

# ── optional babeltrace2 import (graceful fallback for offline dev) ──────────
try:
    import bt2
    BT2_AVAILABLE = True
except ImportError:
    BT2_AVAILABLE = False
    print("[WARNING] babeltrace2 (bt2) not found — only NPZ-load path available.",
          file=sys.stderr)

# Add project root to path so we can import existing LMAT modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.Dictionary import Dictionary

###############################################################################
# Logging helper  (stdout so python -u flushes live to SLURM log)
###############################################################################

_JOB_START = time()

def log(msg: str, *, prefix: str = "INFO") -> None:
    """Print a timestamped line to stdout (flushed immediately via python -u)."""
    elapsed = timedelta(seconds=int(time() - _JOB_START))
    now     = datetime.now().strftime("%H:%M:%S")
    print(f"[{now} +{elapsed}] [{prefix}] {msg}", flush=True)

###############################################################################
# Constants
###############################################################################

# Special token IDs (must match Dictionary defaults)
START_TOK = 2
END_TOK   = 3
TRUNC_TOK = 4


###############################################################################
# Argument parsing
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess SockShop LTTng traces → ML-ready NPZ shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--trace_root", required=True,
        help="Root directory that contains the collected trace runs "
             "(e.g. micro-service-trace-data/traces)",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to write preprocessed NPZ shards and vocab files",
    )
    p.add_argument(
        "--splits", nargs="+", default=None,
        metavar="NAME:DIRS:LABEL",
        help="One entry per dataset split.  Format: "
             "'split_name:run_dir1,run_dir2,...:label'  "
             "where label is 0 (normal) or 1 (anomaly).  "
             "The FIRST split entry is treated as the TRAINING split "
             "(used to build vocabulary & latency boundaries).",
    )
    p.add_argument(
        "--split_preset", choices=["five_run_ood"], default=None,
        help="Optional helper that auto-builds split specs for a standard "
             "5-run SockShop dataset. Useful when trace_root contains "
             "normal/, anomaly_cpu/, anomaly_disk/, anomaly_mem/, anomaly_net/.",
    )
    p.add_argument(
        "--seg_mode", choices=["ust", "time"], default="ust",
        help="Segmentation strategy: 'ust' uses OTel span boundaries from "
             "the UST trace; 'time' uses fixed-duration TID windows (fallback).",
    )
    p.add_argument(
        "--window_ms", type=float, default=100.0,
        help="[time mode] Window duration in milliseconds per TID (default 100)",
    )
    p.add_argument(
        "--warmup_s", type=float, default=5.0,
        help="Seconds of trace to skip at the start of each run (warm-up)",
    )
    p.add_argument(
        "--min_events", type=int, default=8,
        help="Minimum number of kernel events in a window (shorter windows discarded)",
    )
    p.add_argument(
        "--max_seq_len", type=int, default=512,
        help="Maximum sequence length; longer sequences are truncated",
    )
    p.add_argument(
        "--n_categories", type=int, default=6, choices=[4, 6, 8, 10],
        help="Number of latency categories (including the 0=pad category)",
    )
    p.add_argument(
        "--shard_size", type=int, default=5000,
        help="Max number of sequences per NPZ shard file",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (for any stochastic steps)",
    )

    p.add_argument(
        "--split_ratios", type=str, default=None,
        metavar="TRAIN:VALID:TEST",
        help="Split the first normal run into train/valid/test after preprocessing, e.g. '0.70:0.15:0.15'.",
    )

    p.add_argument(
        "--txt_dump_dir", type=str, default=None,
        help="Directory containing pre-converted babeltrace2 text files "
             "(produced by 'babeltrace2 <kernel_ctf> > file.txt').  "
             "When a matching .txt file exists for a run, it is parsed "
             "directly instead of using the slow Python bt2 API.  "
             "File naming convention: {run_rel_with_slashes_as_underscores}_kernel.txt "
             "e.g. normal_run01_kernel.txt",
    )
    p.add_argument(
        "--load_vocab", type=str, default=None,
        metavar="PREPROCESSED_DIR",
        help="Load frozen vocab.pkl + delay_spans.pkl from this directory "
             "instead of building them.  Use for all split jobs after the "
             "vocab-build job completes.",
    )
    p.add_argument(
        "--vocab_only", action="store_true", default=False,
        help="Only build and save vocab.pkl + delay_spans.pkl, then exit. "
             "No NPZ shards are written.  Use this for the dedicated vocab-build "
             "job; all split jobs (including train) then use --load_vocab.",
    )
    p.add_argument(
        "--ust_event_name", type=str, default="otel.spans",
        help="LTTng Python event name used by the OTel relay",
    )
    p.add_argument("--normal_dir", type=str, default="normal",
                   help="Scenario directory name for normal runs")
    p.add_argument("--cpu_dir", type=str, default="anomaly_cpu",
                   help="Scenario directory name for CPU anomaly runs")
    p.add_argument("--disk_dir", type=str, default="anomaly_disk",
                   help="Scenario directory name for disk anomaly runs")
    p.add_argument("--mem_dir", type=str, default="anomaly_mem",
                   help="Scenario directory name for memory anomaly runs")
    p.add_argument("--net_dir", type=str, default="anomaly_net",
                   help="Scenario directory name for network anomaly runs")
    p.add_argument("--normal_train_runs", type=str, default="run01,run02,run03",
                   help="Comma-separated normal runs used for train_id")
    p.add_argument("--normal_valid_runs", type=str, default="run04",
                   help="Comma-separated normal runs used for valid_id")
    p.add_argument("--normal_test_runs", type=str, default="run05",
                   help="Comma-separated normal runs used for test_id")
    p.add_argument("--ood_valid_runs", type=str, default="run04",
                   help="Comma-separated anomaly runs used for valid_ood_* splits")
    p.add_argument("--ood_test_runs", type=str, default="run05",
                   help="Comma-separated anomaly runs used for test_ood_* splits")
    return p.parse_args()


def _csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _join_runs(scenario_dir: str, runs: list[str]) -> list[str]:
    return [f"{scenario_dir}/{run}" for run in runs]


def build_split_specs(args) -> list[dict]:
    """Build split specs either from explicit --splits or a preset."""
    if args.split_preset:
        normal_train = _join_runs(args.normal_dir, _csv_list(args.normal_train_runs))
        normal_valid = _join_runs(args.normal_dir, _csv_list(args.normal_valid_runs))
        normal_test = _join_runs(args.normal_dir, _csv_list(args.normal_test_runs))
        ood_valid_runs = _csv_list(args.ood_valid_runs)
        ood_test_runs = _csv_list(args.ood_test_runs)

        split_specs = [
            {"name": "train_id", "dirs": normal_train, "label": 0},
            {"name": "valid_id", "dirs": normal_valid, "label": 0},
            {"name": "test_id", "dirs": normal_test, "label": 0},
            {"name": "valid_ood_cpu", "dirs": _join_runs(args.cpu_dir, ood_valid_runs), "label": 1},
            {"name": "test_ood_cpu", "dirs": _join_runs(args.cpu_dir, ood_test_runs), "label": 1},
            {"name": "valid_ood_disk", "dirs": _join_runs(args.disk_dir, ood_valid_runs), "label": 1},
            {"name": "test_ood_disk", "dirs": _join_runs(args.disk_dir, ood_test_runs), "label": 1},
            {"name": "valid_ood_mem", "dirs": _join_runs(args.mem_dir, ood_valid_runs), "label": 1},
            {"name": "test_ood_mem", "dirs": _join_runs(args.mem_dir, ood_test_runs), "label": 1},
            {"name": "valid_ood_net", "dirs": _join_runs(args.net_dir, ood_valid_runs), "label": 1},
            {"name": "test_ood_net", "dirs": _join_runs(args.net_dir, ood_test_runs), "label": 1},
        ]
        return split_specs

    if not args.splits:
        raise ValueError("Provide either --splits or --split_preset.")

    split_specs = []
    for spec in args.splits:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid split spec '{spec}' — expected NAME:DIRS:LABEL")
        name, dirs_str, label_str = parts
        dirs = [d.strip() for d in dirs_str.split(",") if d.strip()]
        label = int(label_str)
        split_specs.append({"name": name, "dirs": dirs, "label": label})
    return split_specs


def resolve_run_dir(trace_root: str, run_rel: str) -> str:
    """Resolve a run directory under either trace_root/ or trace_root/traces/."""
    candidates = [
        os.path.join(trace_root, run_rel),
        os.path.join(trace_root, "traces", run_rel),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def find_text_dump(txt_dump_dir: str | None, run_dir: str) -> str | None:
    """Find a pre-converted kernel text dump across the old and new layouts."""
    if not txt_dump_dir:
        return None

    parts = run_dir.replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2:
        return None
    scenario = parts[-2]
    run_name = parts[-1]
    flattened = f"{scenario}_{run_name}_kernel.txt"

    candidates = [
        os.path.join(txt_dump_dir, scenario, run_name, "kernel.txt"),
        os.path.join(txt_dump_dir, flattened),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


###############################################################################
# UST span extraction
###############################################################################

def _extract_ust_spans(ust_trace_dir, ust_event_name, warmup_ns):
    """Return a list of (start_ns, end_ns, op) from the UST OTel trace.

    The OTel→LTTng relay writes Python log events containing fields like:
        message = "op=GET /carts/{id}/items trace_id=abc span_id=xyz ..."
    We parse the timestamp as the *end* of the span.  To estimate the start
    we look for matching span_id pairs or fall back to treating each event
    as a zero-duration boundary (the important thing is the time ordering).

    Returns list of dicts: {start_ns, end_ns, op, span_id, trace_id}
    """
    if not BT2_AVAILABLE:
        return []

    # Find the binary trace inside the UST sub-dir
    # Typical path: ust/ust/uid/1002/64-bit/
    ust_ctf_path = _find_ctf_root(ust_trace_dir)
    if ust_ctf_path is None:
        return []

    spans_raw = {}  # span_id → dict with partial info
    spans_out = []

    try:
        it = bt2.TraceCollectionMessageIterator(ust_ctf_path)
        for msg in it:
            if not isinstance(msg, bt2._EventMessageConst):
                continue
            ev = msg.event
            ts = msg.default_clock_snapshot.ns_from_origin

            if ts < warmup_ns:
                continue

            # Match only our OTel relay events
            if ust_event_name not in ev.name:
                continue

            # The relay stores the payload in a 'message' or similar field
            # Try common field names used by lttng-ust Python handler
            raw_msg = None
            for field_name in ("message", "msg", "payload", "asctime"):
                try:
                    raw_msg = str(ev[field_name])
                    break
                except (KeyError, TypeError):
                    continue

            if raw_msg is None:
                # fallback: try first payload field
                try:
                    raw_msg = str(next(iter(ev.payload_field.values())))
                except Exception:
                    continue

            # Parse key=value pairs from the relay log line
            kv = {}
            for token in raw_msg.split():
                if "=" in token:
                    k, _, v = token.partition("=")
                    kv[k] = v

            op       = kv.get("op", "unknown")
            span_id  = kv.get("span_id", None)
            kind     = kv.get("kind", "unknown")    # SERVER / CLIENT / INTERNAL

            if span_id is None:
                span_id = f"auto_{len(spans_out)}"

            if span_id not in spans_raw:
                spans_raw[span_id] = {"span_id": span_id, "op": op,
                                      "kind": kind, "start_ns": ts}
            else:
                # Second event for same span_id → treat as end
                info = spans_raw.pop(span_id)
                if ts > info["start_ns"]:
                    info["end_ns"] = ts
                else:
                    info["end_ns"] = info["start_ns"]
                    info["start_ns"] = ts
                spans_out.append(info)

    except Exception as exc:
        print(f"[WARNING] UST trace parsing failed: {exc}", file=sys.stderr)

    # Any spans never 'closed' → treat as point events
    for info in spans_raw.values():
        info["end_ns"] = info["start_ns"]
        spans_out.append(info)

    # Sort by start time
    spans_out.sort(key=lambda s: s["start_ns"])
    return spans_out


###############################################################################
# Fast text-file kernel event parser (reads babeltrace2 CLI output)
###############################################################################

# babeltrace2 pretty-print format (default sink.text.pretty):
# [HH:MM:SS.nnnnnnnnn] (+delta) hostname event_name: { cpu_id = N }, { field = val, ... }
# Example:
# [17:38:28.431685497] (+0.000000023) compute1 kernel:syscall_entry_read: { cpu_id = 2 }, { tid = 4567, pid = 4567, procname = "nginx" }

import re

# Compiled once at module load — matches the babeltrace2 pretty output line
_BT2_LINE_RE = re.compile(
    r'^\[\d+:\d+:\d+\.(?P<ns>\d+)\]'   # timestamp (we only need nanosecond part for ordering)
    r'.*? (?P<event>[\w:]+):'           # event name (e.g. kernel:syscall_entry_read)
    r'.*?tid = (?P<tid>\d+)'           # tid
    r'.*?pid = (?P<pid>\d+)'           # pid
    r'(?:.*?procname = "(?P<proc>[^"]+)")?'  # procname (optional)
    r'(?:.*?ret = (?P<ret>-?\d+))?',   # ret (optional)
    re.DOTALL,
)

# Faster split-based parser (no regex) — used in hot path
_FIELD_RE = re.compile(r'(\w+) = (?:"([^"]*)"|(-?\d+))')


def _kernel_events_from_text(txt_path: str, warmup_s: float):
    """Parse a babeltrace2 CLI text dump and yield normalised event dicts.

    This is ~5-10x faster than the Python bt2 API because:
    - No per-event Python object allocation from C extensions
    - Bulk file I/O instead of bt2 iterator callbacks
    - We parse only the fields we actually need

    The text file must have been produced by:
        babeltrace2 <kernel_ctf_dir> > output.txt

    Yields the same dict structure as _kernel_events().
    """
    log(f"[TEXT] Reading pre-converted text: {txt_path}", prefix="PARSE")
    t0 = time()
    _count   = 0
    _skipped = 0
    _LOG_EVERY = 1_000_000   # log every 1M lines

    # Warm-up tracking
    first_abs_ts = None
    warmup_ns    = int(warmup_s * 1e9)

    # ── Timestamp strategy ───────────────────────────────────────────────────
    # babeltrace2 pretty output shows HH:MM:SS.nnnnnnnnn which WRAPS at
    # midnight and also loses the date. For a multi-hour trace that starts
    # near midnight this would cause non-monotonic timestamps and break
    # TID-window segmentation.
    #
    # Fix: track the previous decoded second count and detect backward jumps
    # (midnight wrap). Add 86400s of offset each time we see a wrap.
    _prev_sec   = -1
    _day_offset = 0   # seconds added for each midnight crossing

    with open(txt_path, "r", errors="replace") as fh:
        for line in fh:
            _count += 1

            # ── Parse timestamp ─────────────────────────────────────────────
            ts_start = line.find("[")
            ts_end   = line.find("]", ts_start)
            if ts_start < 0 or ts_end < 0:
                continue
            ts_str = line[ts_start+1:ts_end]   # "HH:MM:SS.NNNNNNNNN"
            dot = ts_str.rfind(".")
            if dot < 0:
                continue
            try:
                parts = ts_str[:dot].split(":")
                cur_sec = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                # Detect midnight wrap: if current second is much less than
                # previous (e.g. jumped from 86399 back to 0)
                if _prev_sec >= 0 and cur_sec < _prev_sec - 3600:
                    _day_offset += 86400
                _prev_sec = cur_sec
                sec = cur_sec + _day_offset
                nsec_part = int(ts_str[dot+1:].ljust(9, '0'))
                ns = sec * 1_000_000_000 + nsec_part
            except (ValueError, IndexError):
                continue

            # ── Warm-up skip ────────────────────────────────────────────────
            if first_abs_ts is None:
                first_abs_ts = ns
            if ns - first_abs_ts < warmup_ns:
                _skipped += 1
                if _count % _LOG_EVERY == 0:
                    rate = _count / max(1, time() - t0)
                    log(f"  text read: {_count:>12,}  (skipping warm-up)  "
                        f"{rate:,.0f} lines/s", prefix="PARSE")
                continue

            # ── Extract event name ──────────────────────────────────────────
            after_delta = line.find(")", ts_end)
            if after_delta < 0:
                continue
            rest = line[after_delta+2:]
            sp1 = rest.find(" ")
            sp2 = rest.find(" ", sp1+1)
            if sp1 < 0 or sp2 < 0:
                continue
            raw_name = rest[sp1+1:sp2].rstrip(":")

            name     = raw_name.replace("kernel:", "")
            name     = name.replace("syscall_", "").replace("entry_", "").replace("exit_", "")
            is_entry = "entry" in raw_name
            is_exit  = "exit"  in raw_name

            # ── Extract fields (tid, pid, procname, ret) ────────────────────
            payload = rest[sp2:]
            pid, tid, procname, ret_enc = 0, 0, "unknown", 0
            for m in _FIELD_RE.finditer(payload):
                fname = m.group(1)
                sval  = m.group(2)
                ival  = m.group(3)
                if fname == "tid" and ival:
                    tid = int(ival)
                elif fname == "pid" and ival:
                    pid = int(ival)
                elif fname == "procname" and sval:
                    procname = sval
                elif fname == "ret" and ival:
                    v = int(ival)
                    ret_enc = 1 if v >= 0 else 2

            if _count % _LOG_EVERY == 0:
                elapsed = timedelta(seconds=int(time() - t0))
                kept    = _count - _skipped
                rate    = _count / max(1, time() - t0)
                log(f"  text read: {_count:>12,} total  {kept:>12,} kept  "
                    f"{_skipped:>8,} warm-up  {rate:,.0f} lines/s  "
                    f"elapsed {elapsed}", prefix="PARSE")

            yield {
                "name":      name,
                "raw_name":  raw_name,
                "timestamp": ns,
                "pid":       pid,
                "tid":       tid,
                "procname":  procname,
                "ret":       ret_enc,
                "is_entry":  is_entry,
                "is_exit":   is_exit,
            }

    elapsed = timedelta(seconds=int(time() - t0))
    rate    = _count / max(1, time() - t0)
    log(f"  text parse complete: {_count:,} lines  {rate:,.0f} lines/s  "
        f"elapsed {elapsed}", prefix="PARSE")


###############################################################################
# babeltrace2 Python API kernel event iterator (slow path / fallback)
###############################################################################

def _kernel_events(kernel_trace_dir, warmup_ns):
    """Yield normalised event dicts from the kernel LTTng trace.

    Yields dicts with keys:
        name, timestamp, pid, tid, procname, ret, is_entry, is_exit
    """
    if not BT2_AVAILABLE:
        raise RuntimeError("babeltrace2 (bt2) required for trace parsing.")

    ctf_path = _find_ctf_root(kernel_trace_dir)
    if ctf_path is None:
        raise FileNotFoundError(
            f"No CTF trace found under {kernel_trace_dir}"
        )

    it = bt2.TraceCollectionMessageIterator(ctf_path)
    _counter = 0
    _skipped = 0
    _t0 = time()
    _LOG_EVERY = 500_000

    for msg in it:
        if not isinstance(msg, bt2._EventMessageConst):
            continue
        ev = msg.event
        ts = msg.default_clock_snapshot.ns_from_origin

        _counter += 1
        if ts < warmup_ns:
            _skipped += 1
            if _counter % _LOG_EVERY == 0:
                rate = _counter / max(1, time() - _t0)
                log(f"  kernel read: {_counter:>10,} events  "
                    f"(skipping warm-up)  {rate:,.0f} ev/s",
                    prefix="PARSE")
            continue

        if _counter % _LOG_EVERY == 0:
            elapsed = timedelta(seconds=int(time() - _t0))
            kept    = _counter - _skipped
            rate    = kept / max(1, time() - _t0)
            log(f"  kernel read: {_counter:>10,} total  "
                f"{kept:>10,} kept  {_skipped:>8,} warm-up skipped  "
                f"{rate:,.0f} ev/s  elapsed {elapsed}",
                prefix="PARSE")

        # Normalise syscall name (remove prefixes)
        raw_name = ev.name
        name = raw_name.replace("syscall_", "")
        name = name.replace("entry_", "").replace("exit_", "")

        is_entry = "entry" in raw_name
        is_exit  = "exit"  in raw_name

        pid = _safe_field(ev, ("vpid", "pid"), 0)
        tid = _safe_field(ev, ("vtid", "tid"), 0)
        procname = str(_safe_field(ev, ("procname",), "unknown"))
        ret_raw  = _safe_field(ev, ("ret",), None)
        if ret_raw is None:
            ret_enc = 0
        elif int(ret_raw) >= 0:
            ret_enc = 1  # success
        else:
            ret_enc = 2  # failure

        yield {
            "name":      name,
            "raw_name":  raw_name,
            "timestamp": ts,
            "pid":       int(pid),
            "tid":       int(tid),
            "procname":  procname,
            "ret":       ret_enc,
            "is_entry":  is_entry,
            "is_exit":   is_exit,
        }


def _safe_field(ev, field_names, default):
    for fn in field_names:
        try:
            return ev[fn]
        except (KeyError, TypeError):
            continue
    return default


def _find_ctf_root(base_dir):
    """Walk base_dir to find the directory that contains CTF index files."""
    if not os.path.isdir(base_dir):
        return None
    for root, dirs, files in os.walk(base_dir):
        if any(f.endswith(".idx") for f in files):
            return root
        # Also accept if the dir itself is the CTF stream root
        if any(f.endswith(".ctf") or f == "metadata" for f in files):
            return root
    # fallback: just return base_dir and let bt2 figure it out
    return base_dir


###############################################################################
# Segmentation strategies
###############################################################################

def segment_ust(kernel_events_list, spans, min_events):
    """Match kernel events to UST OTel spans by timestamp overlap.

    For each span [start_ns, end_ns] collect all kernel events whose
    timestamp falls in [start_ns, end_ns].  Return list of event lists.
    """
    # Build a sorted array of (start_ns, end_ns, span_idx) for binary search
    if not spans:
        return []

    starts = np.array([s["start_ns"] for s in spans], dtype=np.int64)
    ends   = np.array([s["end_ns"]   for s in spans], dtype=np.int64)

    # Buckets indexed by span index
    buckets = [[] for _ in spans]

    for ev in kernel_events_list:
        ts = ev["timestamp"]
        # Find all spans that cover this timestamp
        # Spans are sorted by start; find first span with start <= ts
        idx = np.searchsorted(starts, ts, side="right") - 1
        # Check a small neighbourhood (spans may overlap)
        for i in range(max(0, idx - 1), min(len(spans), idx + 3)):
            if starts[i] <= ts <= ends[i]:
                buckets[i].append(ev)

    # Filter short windows
    sequences = []
    for i, bucket in enumerate(buckets):
        if len(bucket) >= min_events:
            span_dur_ms = (ends[i] - starts[i]) / 1e6
            sequences.append((bucket, span_dur_ms, spans[i].get("op", "")))
    return sequences


def _segment_time_streaming(event_gen, window_ms, min_events):
    """TID-based fixed-duration segmentation that consumes a *generator*.

    Identical output to segment_time() but never holds all events in RAM:
    only the per-TID rolling buffers are kept in memory (order of tens of
    thousands of events across all active TIDs at once, not 500M).

    Returns list of (events, duration_ms, label_str).
    """
    window_ns    = int(window_ms * 1e6)
    tid_buffers  = defaultdict(list)
    tid_starts   = {}
    sequences    = []
    _count       = 0
    _t0          = time()
    _LOG_EVERY   = 5_000_000   # log every 5M events during segmentation

    for ev in event_gen:
        _count += 1
        tid = ev["tid"]
        ts  = ev["timestamp"]

        if tid not in tid_starts:
            tid_starts[tid] = ts

        elapsed = ts - tid_starts[tid]
        if elapsed >= window_ns and len(tid_buffers[tid]) >= 1:
            buf = tid_buffers[tid]
            dur_ms = elapsed / 1e6
            if len(buf) >= min_events:
                sequences.append((list(buf), dur_ms, f"tid_{tid}"))
            tid_buffers[tid] = [ev]
            tid_starts[tid]  = ts
        else:
            tid_buffers[tid].append(ev)

        if _count % _LOG_EVERY == 0:
            elapsed_wall = timedelta(seconds=int(time() - _t0))
            rate = _count / max(1, time() - _t0)
            log(f"  streaming seg: {_count:>12,} events read  "
                f"{len(sequences):>7,} seqs so far  "
                f"{rate:,.0f} ev/s  elapsed {elapsed_wall}",
                prefix="SEG")

    # Flush remaining partial windows
    for tid, buf in tid_buffers.items():
        if len(buf) >= min_events and tid in tid_starts:
            dur_ms = (buf[-1]["timestamp"] - tid_starts[tid]) / 1e6
            sequences.append((list(buf), dur_ms, f"tid_{tid}"))

    total_wall = timedelta(seconds=int(time() - _t0))
    log(f"  streaming seg done: {_count:,} events → {len(sequences):,} seqs  "
        f"elapsed {total_wall}", prefix="SEG")
    return sequences


def segment_time(kernel_events_list, window_ms, min_events):
    """TID-based fixed-duration sliding window segmentation.

    Returns list of (events, duration_ms, label_str).
    """
    if not kernel_events_list:
        return []

    window_ns = int(window_ms * 1e6)
    tid_buffers  = defaultdict(list)
    tid_starts   = {}
    sequences    = []

    for ev in kernel_events_list:
        tid = ev["tid"]
        ts  = ev["timestamp"]

        if tid not in tid_starts:
            tid_starts[tid] = ts

        elapsed = ts - tid_starts[tid]
        if elapsed >= window_ns and len(tid_buffers[tid]) >= 1:
            # Emit the current window
            buf = tid_buffers[tid]
            dur_ms = elapsed / 1e6
            if len(buf) >= min_events:
                sequences.append((list(buf), dur_ms, f"tid_{tid}"))
            # Start new window
            tid_buffers[tid] = [ev]
            tid_starts[tid]  = ts
        else:
            tid_buffers[tid].append(ev)

    # Flush remaining partial windows
    for tid, buf in tid_buffers.items():
        if len(buf) >= min_events and tid in tid_starts:
            dur_ms = (buf[-1]["timestamp"] - tid_starts[tid]) / 1e6
            sequences.append((list(buf), dur_ms, f"tid_{tid}"))

    return sequences


###############################################################################
# Sequence encoding
###############################################################################

def encode_sequence(events, dict_sys, dict_proc, is_train, max_seq_len):
    """Convert a list of event dicts into parallel integer arrays.

    Returns dict with keys: call, entry, duration, proc, pid, tid, ret,
    raw_latency, event_names
    Plus req_dur_ms is set from outside.
    """
    # Start token
    call  = [START_TOK]
    entry = [0]
    dur   = [0]
    proc  = [dict_proc.get_idx("[START]")]
    pid   = [0]
    tid   = [0]
    ret   = [0]
    raw_lat = [0]          # raw latency in ns (for categorisation later)
    ev_names = [""]        # event name (for categorisation later)

    # Running state for entry→exit latency
    entry_time_map = {}    # (name, tid) → entry_timestamp

    prev_ts = None

    for ev in events:
        name     = ev["name"]
        ts       = ev["timestamp"]
        is_entry = ev["is_entry"]
        is_exit  = ev["is_exit"]

        sysname  = name
        procname = ev["procname"]

        if is_train:
            dict_sys.add_word(sysname)
            dict_proc.add_word(procname)

        call.append(dict_sys.get_idx(sysname))
        proc.append(dict_proc.get_idx(procname))

        entry_val = 1 if is_entry else (2 if is_exit else 0)
        entry.append(entry_val)

        delta = (ts - prev_ts) if prev_ts is not None else 0
        dur.append(max(0, delta))
        prev_ts = ts

        pid.append(ev["pid"])
        tid.append(ev["tid"])
        ret.append(ev["ret"])

        # Compute latency (duration of the entry→exit pair)
        key = (sysname, ev["tid"])
        if is_entry:
            entry_time_map[key] = ts
            lat = 0
        elif is_exit and key in entry_time_map:
            lat = ts - entry_time_map.pop(key)
            lat = max(0, lat)
        else:
            lat = 0
        raw_lat.append(lat)
        ev_names.append(ev["raw_name"])

    # End token
    call.append(END_TOK)
    proc.append(dict_proc.get_idx("[END]"))
    entry.append(0); dur.append(0); pid.append(0); tid.append(0)
    ret.append(0);   raw_lat.append(0); ev_names.append("")

    # Truncate
    if len(call) > max_seq_len:
        call     = call[:max_seq_len - 1]  + [TRUNC_TOK]
        entry    = entry[:max_seq_len]
        dur      = dur[:max_seq_len]
        proc     = proc[:max_seq_len - 1]  + [TRUNC_TOK]
        pid      = pid[:max_seq_len]
        tid      = tid[:max_seq_len]
        ret      = ret[:max_seq_len]
        raw_lat  = raw_lat[:max_seq_len]
        ev_names = ev_names[:max_seq_len]

    return {
        "call":      call,
        "entry":     entry,
        "duration":  dur,
        "proc":      proc,
        "pid":       pid,
        "tid":       tid,
        "ret":       ret,
        "raw_lat":   raw_lat,
        "ev_names":  ev_names,
    }


###############################################################################
# Latency categorisation  (reuses LMAT algorithm, vectorised with searchsorted)
###############################################################################

def build_delay_spans(all_latencies, all_names, n_categories):
    """Compute percentile-based latency boundaries for each event type.

    Parameters
    ----------
    all_latencies : list[list[int]]   — raw latency (ns) per sequence
    all_names     : list[list[str]]   — event names  per sequence
    n_categories  : int               — including the 0=pad category
    n_cat = n_categories - 1          — actual bins

    Returns
    -------
    delay_spans : dict[str, (np.ndarray boundaries, int count)]
    """
    n_cat = n_categories - 1
    total = n_cat * (n_cat + 1) / 2

    event_latencies = defaultdict(list)
    for lats, names in zip(all_latencies, all_names):
        for l, n in zip(lats, names):
            # Only exit events have meaningful latency
            if "exit" in n and l > 0:
                base = n.replace("syscall_", "").replace("exit_", "")
                event_latencies[base].append(l)

    delay_spans = {}
    cumperc = 0.0
    for ev_name, latlist in event_latencies.items():
        if len(latlist) < 2:
            continue
        arr = np.sort(np.array(latlist, dtype=np.float64))
        percentiles = []
        cumperc = 0.0
        for i in range(1, n_cat):
            fraction = (n_cat - i + 1) / total
            cumperc += fraction * 100
            percentiles.append(cumperc)
        boundaries = np.percentile(arr, percentiles)
        delay_spans[ev_name] = (boundaries, len(latlist))
    return delay_spans


def merge_delay_spans(existing, new_spans):
    """Weighted-average merge of latency boundary dicts."""
    merged = dict(existing)
    for ev, (new_bnd, new_cnt) in new_spans.items():
        if ev in merged:
            old_bnd, old_cnt = merged[ev]
            w_avg = (old_bnd * old_cnt + new_bnd * new_cnt) / (old_cnt + new_cnt)
            merged[ev] = (w_avg, old_cnt + new_cnt)
        else:
            merged[ev] = (new_bnd, new_cnt)
    return merged


def categorise_latency_vec(raw_lat, ev_names, delay_spans):
    """Vectorised latency categorisation using np.searchsorted.

    Parameters
    ----------
    raw_lat    : list[int]   — raw latency (ns) per event in one sequence
    ev_names   : list[str]   — event names
    delay_spans: dict

    Returns
    -------
    np.ndarray of uint8, shape (len(raw_lat),)
    """
    cats = np.zeros(len(raw_lat), dtype=np.uint8)
    for i, (l, n) in enumerate(zip(raw_lat, ev_names)):
        if "exit" not in n or l == 0:
            cats[i] = 0
            continue
        base = n.replace("syscall_", "").replace("exit_", "")
        info = delay_spans.get(base)
        if info is None:
            cats[i] = 0
            continue
        boundaries, _ = info
        # searchsorted(side='left') gives index of first boundary > l
        # category = index + 1  (1-indexed), capped at n_cat
        cat = int(np.searchsorted(boundaries, l, side="left")) + 1
        cats[i] = min(cat, len(boundaries) + 1)
    return cats


###############################################################################
# NPZ shard writing
###############################################################################

def write_shards(sequences, split_dir, shard_size, delay_spans, is_anomaly):
    """Pad sequences within each shard and write NPZ files.

    Parameters
    ----------
    sequences  : list of dicts with keys call, entry, duration, proc,
                 pid, tid, ret, lat_cat (np.ndarray), req_dur_ms, seq_len
    split_dir  : output directory path
    shard_size : sequences per shard
    delay_spans: for reference (latency boundaries already applied)
    is_anomaly : int, 0 or 1
    """
    os.makedirs(split_dir, exist_ok=True)
    n_total  = len(sequences)
    n_shards = (n_total + shard_size - 1) // shard_size
    written  = 0
    t0_write = time()

    log(f"Writing {n_total:,} sequences → {n_shards} shard(s) in {split_dir}",
        prefix="SHARD")

    for shard_idx in range(n_shards):
        batch = sequences[shard_idx * shard_size:(shard_idx + 1) * shard_size]
        if not batch:
            break

        t_shard = time()
        max_len = max(s["seq_len"] for s in batch)

        def pad(arr, maxl, pad_val=0, dtype=np.int32):
            out = np.full((len(batch), maxl), pad_val, dtype=dtype)
            for i, s in enumerate(batch):
                v = s[arr]
                out[i, :len(v)] = v
            return out

        shard_data = {
            "call":       pad("call",     max_len, 0, np.int32),
            "entry":      pad("entry",    max_len, 0, np.int8),
            "duration":   pad("duration", max_len, 0, np.int64),
            "proc":       pad("proc",     max_len, 0, np.int32),
            "pid":        pad("pid",      max_len, 0, np.int32),
            "tid":        pad("tid",      max_len, 0, np.int32),
            "ret":        pad("ret",      max_len, 0, np.int8),
            "lat_cat":    pad("lat_cat",  max_len, 0, np.uint8),
            "seq_len":    np.array([s["seq_len"]   for s in batch], dtype=np.int32),
            "req_dur_ms": np.array([s["req_dur_ms"] for s in batch], dtype=np.float32),
            "is_anomaly": np.full(len(batch), is_anomaly, dtype=np.int8),
        }

        shard_path = os.path.join(split_dir, f"shard_{shard_idx:06d}.npz")
        np.savez_compressed(shard_path, **shard_data)
        written += len(batch)

        # ── progress line with ETA ───────────────────────────────────────────
        pct      = written / n_total * 100
        elapsed  = time() - t0_write
        eta_s    = (elapsed / written) * (n_total - written) if written else 0
        eta_str  = str(timedelta(seconds=int(eta_s)))
        shard_ms = (time() - t_shard) * 1000
        log(f"  shard {shard_idx+1:>4}/{n_shards}  "
            f"{written:>7,}/{n_total:,} seqs  ({pct:5.1f}%)  "
            f"padded_len={max_len}  {shard_ms:.0f}ms/shard  ETA {eta_str}",
            prefix="SHARD")

    return written


###############################################################################
# Per-run processing
###############################################################################

def process_run(run_dir, seg_mode, window_ms, warmup_s, min_events,
                max_seq_len, n_categories, dict_sys, dict_proc, is_train,
                delay_spans, is_anomaly, ust_event_name, txt_dump_dir=None):
    """Process one LTTng run directory.

    Returns (sequences_with_latcat, updated_delay_spans)
    where sequences_with_latcat is a list of encoded sequence dicts.

    If txt_dump_dir is set and a pre-converted text file for this run exists,
    the fast text-file parser is used instead of the Python bt2 API.
    """
    kernel_dir = os.path.join(run_dir, "kernel")
    ust_dir    = os.path.join(run_dir, "ust")

    warmup_ns = int(warmup_s * 1e9)

    log(f"{'─'*56}", prefix="RUN")
    log(f"Run dir  : {run_dir}", prefix="RUN")
    log(f"seg_mode={seg_mode}  warmup={warmup_s}s  "
        f"window_ms={window_ms}  is_train={is_train}", prefix="RUN")

    # ── 1. Collect kernel events ─────────────────────────────────────────────
    # Determine whether a pre-converted text file is available (fast path)
    txt_file = find_text_dump(txt_dump_dir, run_dir)
    if txt_dump_dir:
        if txt_file:
            log(f"Text dump found: {txt_file}", prefix="PARSE")
        else:
            log(f"No text dump found for {run_dir} — falling back to bt2 API",
                prefix="WARN")

    t0 = time()
    kernel_evs = []

    if txt_file:
        # ── Fast path: read pre-converted text file ──────────────────────────
        # IMPORTANT: we collect into a list here because segment_time needs
        # random access (TID-keyed buffers).  For a 500M-event trace this is
        # ~150GB of Python dicts which will OOM.  Instead, run segmentation
        # *streaming* directly from the generator.
        log(f"Streaming text → segmentation (no full in-memory list) ...",
            prefix="PARSE")
        event_gen = _kernel_events_from_text(txt_file, warmup_s)
    elif BT2_AVAILABLE and os.path.isdir(kernel_dir):
        # ── Slow path: Python bt2 API ────────────────────────────────────────
        log(f"Reading kernel trace via bt2 from {kernel_dir} ...", prefix="PARSE")
        def _bt2_gen():
            first_ts = None
            for ev in _kernel_events(kernel_dir, warmup_ns=0):
                if first_ts is None:
                    first_ts = ev["timestamp"]
                if ev["timestamp"] - (first_ts or 0) < int(warmup_s * 1e9):
                    continue
                yield ev
        event_gen = _bt2_gen()
    else:
        log(f"[SKIP] kernel dir not found or bt2 unavailable: {kernel_dir}",
            prefix="WARN")
        return [], delay_spans

    # ── 2. Segment (streaming — never holds all events in RAM) ───────────────
    t0_seg = time()
    if seg_mode == "time":
        log(f"TID time-window segmentation (window_ms={window_ms}, streaming) ...",
            prefix="SEG")
        raw_segments = _segment_time_streaming(event_gen, window_ms, min_events)
    else:
        # UST mode still needs full list for binary search; collect first.
        log(f"Collecting events for UST segmentation ...", prefix="SEG")
        kernel_evs = list(event_gen)
        parse_dur = timedelta(seconds=round(time() - t0))
        log(f"Kernel events collected: {len(kernel_evs):,}  "
            f"(parse time {parse_dur})", prefix="PARSE")
        if not kernel_evs:
            log("[SKIP] No kernel events.", prefix="WARN")
            return [], delay_spans
        trace_start_ns = kernel_evs[0]["timestamp"]
        if os.path.isdir(ust_dir):
            log(f"Extracting UST OTel spans from {ust_dir} ...", prefix="SEG")
            warmup_abs_ns = trace_start_ns + int(warmup_s * 1e9)
            spans = _extract_ust_spans(ust_dir, ust_event_name, warmup_abs_ns)
            log(f"UST spans extracted: {len(spans):,}", prefix="SEG")
            if spans:
                raw_segments = segment_ust(kernel_evs, spans, min_events)
            else:
                log("No UST spans — falling back to time-window", prefix="WARN")
                raw_segments = segment_time(kernel_evs, window_ms, min_events)
        else:
            raw_segments = segment_time(kernel_evs, window_ms, min_events)
        kernel_evs = []   # free memory immediately

    seg_dur = timedelta(seconds=round(time() - t0_seg))
    log(f"Segments produced: {len(raw_segments):,}  "
        f"(min_events={min_events}  seg time {seg_dur})", prefix="SEG")

    parse_dur = timedelta(seconds=round(time() - t0))
    log(f"Parse+segment total time: {parse_dur}", prefix="PARSE")

    if not raw_segments:
        log("[SKIP] Zero segments after filtering — check --min_events.",
            prefix="WARN")
        return [], delay_spans

    # ── 3. Encode sequences ─────────────────────────────────────────────────
    t0 = time()
    log(f"Encoding {len(raw_segments):,} sequences ...", prefix="ENC")
    encoded       = []
    all_latencies = []
    all_names     = []
    _LOG_ENC = max(1, len(raw_segments) // 20)  # log ~20 times

    for idx, (evs, dur_ms, _) in enumerate(raw_segments):
        enc = encode_sequence(evs, dict_sys, dict_proc, is_train, max_seq_len)
        enc["req_dur_ms"] = dur_ms
        enc["seq_len"]    = len(enc["call"])
        encoded.append(enc)
        all_latencies.append(enc["raw_lat"])
        all_names.append(enc["ev_names"])

        if (idx + 1) % _LOG_ENC == 0 or (idx + 1) == len(raw_segments):
            pct     = (idx + 1) / len(raw_segments) * 100
            elapsed = time() - t0
            eta_s   = (elapsed / (idx + 1)) * (len(raw_segments) - idx - 1)
            log(f"  encoded {idx+1:>7,}/{len(raw_segments):,}  "
                f"({pct:5.1f}%)  vocab={len(dict_sys)}  "
                f"ETA {timedelta(seconds=int(eta_s))}",
                prefix="ENC")

    enc_dur = timedelta(seconds=round(time() - t0))
    log(f"Encoding done in {enc_dur}  "
        f"vocab: {len(dict_sys)} syscalls / {len(dict_proc)} procs",
        prefix="ENC")

    # ── 4. Update latency boundaries (train only) ────────────────────────────
    if is_train:
        log("Building latency boundary spans from training data ...", prefix="LAT")
        t0 = time()
        new_spans = build_delay_spans(all_latencies, all_names, n_categories)
        if delay_spans is None:
            delay_spans = new_spans
        else:
            delay_spans = merge_delay_spans(delay_spans, new_spans)
        log(f"Delay spans updated: {len(delay_spans)} event types  "
            f"({timedelta(seconds=round(time()-t0))})", prefix="LAT")

    # ── 5. Categorise latencies ─────────────────────────────────────────────
    t0 = time()
    if delay_spans is not None:
        log(f"Categorising latencies for {len(encoded):,} sequences ...",
            prefix="LAT")
        for enc in encoded:
            enc["lat_cat"] = categorise_latency_vec(
                enc["raw_lat"], enc["ev_names"], delay_spans
            )
        log(f"Latency categorisation done  ({timedelta(seconds=round(time()-t0))})",
            prefix="LAT")
    else:
        log("No delay spans yet — lat_cat set to zeros (will be re-applied later)",
            prefix="WARN")
        for enc in encoded:
            enc["lat_cat"] = np.zeros(enc["seq_len"], dtype=np.uint8)

    return encoded, delay_spans


###############################################################################
# Main
###############################################################################

def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.split_preset and args.split_ratios:
        log("--split_preset and --split_ratios are mutually exclusive; "
            "use explicit run-based splits for the 5-run preset.",
            prefix="ERROR")
        sys.exit(1)
    if args.load_vocab and args.split_ratios:
        log("--load_vocab cannot be combined with --split_ratios because "
            "split_ratios requires rebuilding the training split first.",
            prefix="ERROR")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    log("="*60, prefix="START")
    log("SockShop LTTng → NPZ preprocessing", prefix="START")
    log(f"trace_root  : {args.trace_root}",  prefix="START")
    log(f"output_dir  : {args.output_dir}",  prefix="START")
    log(f"seg_mode    : {args.seg_mode}",     prefix="START")
    log(f"n_categories: {args.n_categories}", prefix="START")
    log(f"shard_size  : {args.shard_size}",   prefix="START")
    log(f"warmup_s    : {args.warmup_s}",     prefix="START")
    log(f"min_events  : {args.min_events}",   prefix="START")
    log("="*60, prefix="START")

    # Parse split specs ---------------------------------------------------
    try:
        split_specs = build_split_specs(args)
    except ValueError as exc:
        log(str(exc), prefix="ERROR")
        sys.exit(1)

    log(f"Splits to process ({len(split_specs)} total):", prefix="START")
    for s in split_specs:
        log(f"  {s['name']:<20} dirs={s['dirs']}  label={s['label']}",
            prefix="START")

    if not split_specs:
        print("[ERROR] No splits specified.", file=sys.stderr)
        sys.exit(1)

    train_spec  = split_specs[0]
    other_specs = split_specs[1:]

    vocab_path = os.path.join(args.output_dir, "vocab.pkl")
    delay_path = os.path.join(args.output_dir, "delay_spans.pkl")

    # ── STEP 1: Training split (builds vocab) OR load frozen vocab ────────────
    if args.load_vocab:
        # ── Fast path: anomaly job — load vocab frozen by the training job ───
        frozen_dir = args.load_vocab
        _vp = os.path.join(frozen_dir, "vocab.pkl")
        _dp = os.path.join(frozen_dir, "delay_spans.pkl")
        if not os.path.isfile(_vp) or not os.path.isfile(_dp):
            log(f"--load_vocab: vocab.pkl or delay_spans.pkl not found in {frozen_dir}",
                prefix="ERROR")
            sys.exit(1)
        with open(_vp, "rb") as f:
            dict_sys, dict_proc = pickle.load(f)
        with open(_dp, "rb") as f:
            delay_spans = pickle.load(f)
        log(f"Loaded frozen vocab from {frozen_dir}  "
            f"({len(dict_sys)} syscalls / {len(dict_proc)} procs)",
            prefix="VOCAB")
        log(f"Loaded delay_spans: {len(delay_spans)} event types", prefix="VOCAB")
        # In anomaly-only mode all supplied splits go to other_specs
        # (there is no training split to process)
        other_specs = split_specs
        train_spec  = None
    else:
        # ── Normal path: process the first (training) split, build vocab ─────
        log(f"\n{'='*60}", prefix="TRAIN")
        log(f"Processing training split '{train_spec['name']}'", prefix="TRAIN")
        log(f"{'='*60}", prefix="TRAIN")

        dict_sys    = Dictionary()
        dict_proc   = Dictionary()
        delay_spans = None
        train_sequences = []

        for run_rel in train_spec["dirs"]:
            run_dir = resolve_run_dir(args.trace_root, run_rel)
            if not os.path.isdir(run_dir):
                log(f"Run dir not found: {run_dir}", prefix="WARN")
                continue
            seqs, delay_spans = process_run(
                run_dir,
                seg_mode       = args.seg_mode,
                window_ms      = args.window_ms,
                warmup_s       = args.warmup_s,
                min_events     = args.min_events,
                max_seq_len    = args.max_seq_len,
                n_categories   = args.n_categories,
                dict_sys       = dict_sys,
                dict_proc      = dict_proc,
                is_train       = True,
                delay_spans    = delay_spans,
                is_anomaly     = train_spec["label"],
                ust_event_name = args.ust_event_name,
                txt_dump_dir   = args.txt_dump_dir,
            )
            train_sequences.extend(seqs)

        log(f"TRAIN total sequences: {len(train_sequences):,}", prefix="TRAIN")

        if not train_sequences:
            log("Training split produced zero sequences — check trace paths.",
                prefix="ERROR")
            sys.exit(1)

        if args.split_ratios:
            ratios = [float(x) for x in args.split_ratios.split(":")]
            if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
                log("--split_ratios must be 3 floats summing to 1.0 e.g. '0.70:0.15:0.15'",
                    prefix="ERROR")
                sys.exit(1)
            if len(split_specs) < 3:
                log("--split_ratios requires 3+ splits (train,valid,test,...)",
                    prefix="ERROR")
                sys.exit(1)

            # Shuffle with fixed seed for reproducibility
            rng = np.random.default_rng(args.seed)
            rng.shuffle(train_sequences)

            n = len(train_sequences)
            n_tr = int(n * ratios[0])
            n_va = int(n * ratios[1])
            splits_data = [
                (split_specs[0], train_sequences[:n_tr]),           # train_id
                (split_specs[1], train_sequences[n_tr:n_tr + n_va]), # valid_id
                (split_specs[2], train_sequences[n_tr + n_va:]),     # test_id
            ]
            log(f"split_ratios={args.split_ratios}  "
                f"train={n_tr:,} valid={n_va:,} test={n-n_tr-n_va:,}",
                prefix="SPLIT")

            # Save vocab/delay_spans ONCE from full dataset
            os.makedirs(args.output_dir, exist_ok=True)
            with open(vocab_path, "wb") as f:
                pickle.dump((dict_sys, dict_proc), f)
            with open(delay_path, "wb") as f:
                pickle.dump(delay_spans, f)
            log(f"Saved vocab → {vocab_path}", prefix="VOCAB")
            log(f"Saved delays → {delay_path}", prefix="VOCAB")

            if args.vocab_only:
                log("--vocab_only: vocab saved, skipping shards.", prefix="DONE")
                return

            # Write the three normal splits
            for spec, seqs in splits_data:
                split_dir = os.path.join(args.output_dir, spec["name"])
                n_written = write_shards(seqs, split_dir, args.shard_size,
                                         delay_spans, spec["label"])
                meta = {
                    "split": spec["name"],
                    "n_sequences": n_written,
                    "n_vocab_sys": len(dict_sys),
                    "n_vocab_proc": len(dict_proc),
                    "n_categories": args.n_categories,
                    "seg_mode": args.seg_mode,
                    "is_anomaly": spec["label"],
                    "dirs": spec["dirs"],
                    "split_ratios": args.split_ratios,
                }
                with open(os.path.join(split_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

            # Skip original train write + advance other_specs to anomalies only
            other_specs = split_specs[3:]
        else:
            # ── Original single-train-split path (unchanged) ──────────────────
            os.makedirs(args.output_dir, exist_ok=True)
            with open(vocab_path, "wb") as f:
                pickle.dump((dict_sys, dict_proc), f)
            with open(delay_path, "wb") as f:
                pickle.dump(delay_spans, f)
            log(f"Vocabulary: {len(dict_sys):,} syscalls / {len(dict_proc):,} procs",
                prefix="VOCAB")
            log(f"Saved vocab → {vocab_path}", prefix="VOCAB")
            log(f"Saved delays → {delay_path}", prefix="VOCAB")

            if args.vocab_only:
                log("--vocab_only set: vocab saved, skipping shard writing.", prefix="DONE")
                return

            train_dir = os.path.join(args.output_dir, train_spec["name"])
            n = write_shards(train_sequences, train_dir, args.shard_size,
                             delay_spans, train_spec["label"])
            meta = {
                "split": train_spec["name"],
                "n_sequences": n,
                "n_vocab_sys": len(dict_sys),
                "n_vocab_proc": len(dict_proc),
                "n_categories": args.n_categories,
                "seg_mode": args.seg_mode,
                "is_anomaly": train_spec["label"],
                "dirs": train_spec["dirs"],
            }
            with open(os.path.join(train_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)


    # STEP 2: Process remaining splits (vocab + delay_spans frozen)
    for spec in other_specs:
        log(f"{'='*60}", prefix="SPLIT")
        log(f"Processing split '{spec['name']}'", prefix="SPLIT")

        split_sequences = []
        for run_rel in spec["dirs"]:
            run_dir = resolve_run_dir(args.trace_root, run_rel)
            if not os.path.isdir(run_dir):
                log(f"Run dir not found: {run_dir}", prefix="WARN")
                continue
            seqs, _ = process_run(
                run_dir,
                seg_mode       = args.seg_mode,
                window_ms      = args.window_ms,
                warmup_s       = args.warmup_s,
                min_events     = args.min_events,
                max_seq_len    = args.max_seq_len,
                n_categories   = args.n_categories,
                dict_sys       = dict_sys,
                dict_proc      = dict_proc,
                is_train       = False,
                delay_spans    = delay_spans,
                is_anomaly     = spec["label"],
                ust_event_name = args.ust_event_name,
                txt_dump_dir   = args.txt_dump_dir,
            )
            split_sequences.extend(seqs)

        log(f"{spec['name'].upper()} total sequences: {len(split_sequences):,}",
            prefix="SPLIT")

        split_dir = os.path.join(args.output_dir, spec["name"])
        n = write_shards(split_sequences, split_dir, args.shard_size,
                         delay_spans, spec["label"])

        meta = {
            "split": spec["name"],
            "n_sequences": n,
            "n_categories": args.n_categories,
            "seg_mode": args.seg_mode,
            "is_anomaly": spec["label"],
            "dirs": spec["dirs"],
        }
        with open(os.path.join(split_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[DONE] Output written to: {args.output_dir}", file=sys.stderr)
    print(f"       vocab.pkl      → {vocab_path}", file=sys.stderr)
    print(f"       delay_spans.pkl → {delay_path}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
