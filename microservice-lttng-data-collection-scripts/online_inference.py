#!/usr/bin/env python3
"""
online_inference.py — LMAT co-located inference overhead measurement
=====================================================================

Simulates LMAT running on the GCP VM alongside SockShop.

Uses a babeltrace2 subprocess to read the growing kernel CTF trace, segments
events into 100ms TID-based windows (same logic as preprocess_sockshop.py),
encodes each window into a tensor, and runs a model forward pass.

Supports two modes:
  --mode sync   — inference blocks the collection loop (worst case)
  --mode async  — inference runs in a separate thread via a queue (parallel)

In --replay mode the script reads an already-finished CTF trace at full speed,
which is useful for testing the script offline without running a live experiment.

Usage (live, during an experiment):
    python3 online_inference.py \
        --model_path  ~/checkpoints/model_best_lstm.pt \
        --vocab_path  ~/adaptive_tracer/micro-service-trace-data/preprocessed/vocab.pkl \
        --delay_spans_path ~/adaptive_tracer/micro-service-trace-data/preprocessed/delay_spans.pkl \
        --trace_dir   ~/traces/lmat_sync/run01/kernel \
        --model_type  lstm \
        --n_hidden 1024 --n_layer 6 \
        --dim_sys 48 --dim_entry 12 --dim_ret 12 \
        --dim_proc 48 --dim_pid 12 --dim_tid 12 \
        --dim_order 12 --dim_time 12 \
        --mode sync \
        --log_file ~/experiments/lmat_sync/run01/inference.log

Usage (offline replay for testing):
    python3 online_inference.py ... --replay

Requirements (on GCP VM):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    (babeltrace2 must already be installed — it is used by the tracing setup)
"""

import argparse
import logging
import math
import os
import pickle
import queue
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Project root on sys.path so we can import LMAT modules ──────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models import LSTM, Transformer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("online_inference")

# ── Special tokens (must match preprocessor) ─────────────────────────────────
START_TOK = 2
END_TOK   = 3
TRUNC_TOK = 4

# ── Field regex: same as preprocess_sockshop.py ──────────────────────────────
_FIELD_RE = re.compile(r'(\w+) = (?:"([^"]*)"|([-]?\d+))')


###############################################################################
# CLI
###############################################################################

def get_args():
    p = argparse.ArgumentParser(
        description="LMAT online inference overhead measurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--model_path",       required=True, help="Path to model_best_lstm.pt checkpoint")
    p.add_argument("--vocab_path",       required=True, help="Path to vocab.pkl")
    p.add_argument("--delay_spans_path", required=True, help="Path to delay_spans.pkl (latency boundaries)")
    p.add_argument("--trace_dir",        required=True, help="Kernel CTF trace directory (e.g. ~/traces/lmat_sync/run01/kernel)")
    p.add_argument("--log_file",         default=None,  help="Path to write per-window inference log (CSV)")

    # Experiment mode
    p.add_argument("--mode",    choices=["sync", "async"], default="sync",
                   help="sync: inference blocks collector; async: inference in separate thread")
    p.add_argument("--replay",  action="store_true",
                   help="Read a finished CTF trace at full speed (for offline testing)")

    # Model architecture (must match training config)
    p.add_argument("--model_type", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--n_hidden",   type=int, default=1024)
    p.add_argument("--n_layer",    type=int, default=6)
    p.add_argument("--n_head",     type=int, default=8)
    p.add_argument("--dropout",    type=float, default=0.0)
    p.add_argument("--dim_sys",    type=int, default=48)
    p.add_argument("--dim_entry",  type=int, default=12)
    p.add_argument("--dim_ret",    type=int, default=12)
    p.add_argument("--dim_proc",   type=int, default=48)
    p.add_argument("--dim_pid",    type=int, default=12)
    p.add_argument("--dim_tid",    type=int, default=12)
    p.add_argument("--dim_order",  type=int, default=12)
    p.add_argument("--dim_time",   type=int, default=12)
    p.add_argument("--n_categories", type=int, default=6)

    # Segmentation
    p.add_argument("--window_ms",  type=float, default=100.0, help="Window duration (ms)")
    p.add_argument("--min_events", type=int,   default=8,     help="Minimum events per window")
    p.add_argument("--max_seq_len", type=int,  default=512)

    # Scoring
    p.add_argument("--lat_score_weight", type=float, default=0.3,
                   help="Weight for latency cross-entropy in anomaly score (0.7*event + 0.3*lat)")

    # Async queue
    p.add_argument("--queue_maxsize", type=int, default=50,
                   help="Max windows buffered in async queue before dropping oldest")
    p.add_argument("--torch_threads", type=int, default=2,
                   help="Number of CPU threads PyTorch may use for inference")

    return p.parse_args()


###############################################################################
# Model loading
###############################################################################

def load_model(args, n_syscall, n_process, device):
    kw = dict(
        n_syscall=n_syscall, n_category=args.n_categories, n_process=n_process,
        n_hidden=args.n_hidden, n_layer=args.n_layer, dropout=args.dropout,
        dim_sys=args.dim_sys, dim_entry=args.dim_entry, dim_ret=args.dim_ret,
        dim_proc=args.dim_proc, dim_pid=args.dim_pid, dim_tid=args.dim_tid,
        dim_order=args.dim_order, dim_time=args.dim_time, dim_f_mean=0,
        train_event=True, train_latency=True, ordinal_latency=False,
    )
    if args.model_type == "lstm":
        model = LSTM(**kw)
    else:
        model = Transformer(n_head=args.n_head, activation="gelu", tfixup=False, **kw)

    state = torch.load(args.model_path, map_location=device, weights_only=True)
    # Handle torch.compile _orig_mod prefix if present
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    log.info("Model loaded: %s  params=%s  device=%s",
             args.model_type.upper(),
             f"{sum(p.numel() for p in model.parameters()):,}",
             device)
    return model


###############################################################################
# Babeltrace2 event stream (live or replay)
###############################################################################

def _find_ctf_root(base_dir):
    """Walk base_dir to find the directory containing CTF index files."""
    for root, dirs, files in os.walk(base_dir):
        if any(f.endswith(".idx") for f in files):
            return root
        if any(f == "metadata" for f in files):
            return root
    return base_dir


def stream_events_from_bt2(trace_dir: str, replay: bool):
    """
    Yield event dicts from babeltrace2 subprocess output.

    In live mode: babeltrace2 may exit at EOF because LTTng hasn't flushed the
    ring buffer yet. We retry in a loop, skipping events already yielded (dedup
    by timestamp). The shell wrapper runs `sudo lttng flush` every 2s so new data
    appears on disk between retries.

    In replay mode (--replay): reads the finished CTF once at full speed and exits.

    Each dict: {name, raw_name, timestamp_ns, pid, tid, procname, ret, is_entry, is_exit}
    """
    ctf_root = _find_ctf_root(trace_dir)
    cmd = ["babeltrace2", "--output-format=text", ctf_root]

    _last_ns    = 0        # highest timestamp yielded so far (dedup checkpoint)
    _retry      = 0
    _RETRY_WAIT = 2.0      # seconds between retries in live mode

    while True:
        log.info("babeltrace2 pass #%d — ctf_root=%s  last_ns=%d", _retry, ctf_root, _last_ns)
        _retry += 1

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        # ── Timestamp state (midnight-wrap logic, same as preprocess_sockshop.py) ──
        _prev_sec   = -1
        _day_offset = 0
        n_new       = 0     # events yielded in this pass (after dedup)

        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue

                # ── Parse timestamp ──────────────────────────────────────────
                ts_start = line.find("[")
                ts_end   = line.find("]", ts_start)
                if ts_start < 0 or ts_end < 0:
                    continue
                ts_str = line[ts_start + 1:ts_end]
                dot = ts_str.rfind(".")
                if dot < 0:
                    continue
                try:
                    parts   = ts_str[:dot].split(":")
                    cur_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    if _prev_sec >= 0 and cur_sec < _prev_sec - 3600:
                        _day_offset += 86400
                    _prev_sec = cur_sec
                    sec  = cur_sec + _day_offset
                    nsec = int(ts_str[dot + 1:].ljust(9, "0"))
                    ns   = sec * 1_000_000_000 + nsec
                except (ValueError, IndexError):
                    continue

                # ── Dedup: skip events already yielded in a previous pass ────
                if ns <= _last_ns:
                    continue

                # ── Parse event name ─────────────────────────────────────────
                after_delta = line.find(")", ts_end)
                if after_delta < 0:
                    continue
                rest = line[after_delta + 2:]
                sp1 = rest.find(" ")
                sp2 = rest.find(" ", sp1 + 1)
                if sp1 < 0 or sp2 < 0:
                    continue
                raw_name = rest[sp1 + 1:sp2].rstrip(":")
                name     = raw_name.replace("kernel:", "")
                name     = name.replace("syscall_", "").replace("entry_", "").replace("exit_", "")
                is_entry = "entry" in raw_name
                is_exit  = "exit"  in raw_name

                # ── Parse fields ─────────────────────────────────────────────
                payload  = rest[sp2:]
                pid, tid, procname, ret_enc = 0, 0, "unknown", 0
                for m in _FIELD_RE.finditer(payload):
                    fname = m.group(1)
                    sval  = m.group(2)
                    ival  = m.group(3)
                    if   fname == "tid"      and ival: tid  = int(ival)
                    elif fname == "pid"      and ival: pid  = int(ival)
                    elif fname == "procname" and sval: procname = sval
                    elif fname == "ret"      and ival:
                        v = int(ival)
                        ret_enc = 1 if v >= 0 else 2

                _last_ns = ns
                n_new   += 1
                yield {
                    "name": name, "raw_name": raw_name, "timestamp_ns": ns,
                    "pid": pid, "tid": tid, "procname": procname,
                    "ret": ret_enc, "is_entry": is_entry, "is_exit": is_exit,
                }
        finally:
            proc.terminate()
            proc.wait()

        if replay:
            break   # single-pass for offline replay

        # Live mode: sleep to allow LTTng to flush more ring-buffer data, then retry.
        # If we got zero new events on this pass and have yielded some previously,
        # the session has likely ended — exit gracefully.
        if n_new == 0 and _last_ns > 0:
            log.info("No new events on pass #%d — session likely ended. Exiting.", _retry)
            break
        if n_new == 0:
            log.info("No events yet on pass #%d — waiting for ring buffer flush ...", _retry)
        else:
            log.info("Pass #%d yielded %d new events (last_ns=%d). Waiting %.1fs for next flush.",
                     _retry, n_new, _last_ns, _RETRY_WAIT)
        time.sleep(_RETRY_WAIT)



###############################################################################
# Window segmentation (TID-based, 100ms)
###############################################################################

class WindowSegmenter:
    """Stateful streaming segmenter. Call feed(event), get completed windows."""

    def __init__(self, window_ms: float, min_events: int):
        self.window_ns  = int(window_ms * 1e6)
        self.min_events = min_events
        self._buffers   = defaultdict(list)   # tid → [event, ...]
        self._starts    = {}                  # tid → start_ns

    def feed(self, ev: dict):
        """Feed one event. Returns a completed (events, dur_ms) window or None."""
        tid = ev["tid"]
        ts  = ev["timestamp_ns"]

        if tid not in self._starts:
            self._starts[tid] = ts

        elapsed = ts - self._starts[tid]
        if elapsed >= self.window_ns and len(self._buffers[tid]) >= 1:
            buf    = self._buffers[tid]
            dur_ms = elapsed / 1e6
            result = None
            if len(buf) >= self.min_events:
                result = (list(buf), dur_ms)
            self._buffers[tid] = [ev]
            self._starts[tid]  = ts
            return result
        else:
            self._buffers[tid].append(ev)
            return None


###############################################################################
# Sequence encoding (mirrors preprocess_sockshop.py)
###############################################################################

def encode_window(events: list, dict_sys, dict_proc, delay_spans: dict,
                  n_categories: int, max_seq_len: int):
    """
    Convert a list of raw event dicts into a model-ready dict of 1-D arrays.
    Returns None if encoding fails.
    """
    call  = [START_TOK]
    entry = [0]
    dur   = [0]
    proc  = [dict_proc.get_idx("[START]")]
    pid   = [0]
    tid   = [0]
    ret   = [0]
    lat   = [0]          # latency category

    entry_time_map = {}  # (name, tid) → entry_ns
    prev_ts = None

    for ev in events:
        name     = ev["name"]
        ts       = ev["timestamp_ns"]
        is_entry = ev["is_entry"]
        is_exit  = ev["is_exit"]

        call.append(dict_sys.get_idx(name))
        proc.append(dict_proc.get_idx(ev["procname"]))

        entry.append(1 if is_entry else (2 if is_exit else 0))

        delta = (ts - prev_ts) if prev_ts is not None else 0
        dur.append(max(0, delta))
        prev_ts = ts

        pid.append(ev["pid"])
        tid.append(ev["tid"])
        ret.append(ev["ret"])

        # Latency category
        key = (name, ev["tid"])
        if is_entry:
            entry_time_map[key] = ts
            lat_val = 0
        elif is_exit and key in entry_time_map:
            raw_lat = max(0, ts - entry_time_map.pop(key))
            if name in delay_spans:
                boundaries = delay_spans[name]
                # delay_spans values may be (boundaries_array, other) tuples
                if isinstance(boundaries, tuple):
                    boundaries = boundaries[0]
                try:
                    cat = int(np.searchsorted(np.asarray(boundaries, dtype=np.float64), raw_lat)) + 1
                    lat_val = min(cat, n_categories - 1)
                except Exception:
                    lat_val = 0
            else:
                lat_val = 0
        else:
            lat_val = 0
        lat.append(lat_val)

    # End token
    call.append(END_TOK)
    proc.append(dict_proc.get_idx("[END]"))
    entry.append(0); dur.append(0); pid.append(0); tid.append(0)
    ret.append(0);   lat.append(0)

    # Truncate
    if len(call) > max_seq_len:
        call  = call[:max_seq_len - 1]  + [TRUNC_TOK]
        proc  = proc[:max_seq_len - 1]  + [TRUNC_TOK]
        entry = entry[:max_seq_len]
        dur   = dur[:max_seq_len]
        pid   = pid[:max_seq_len]
        tid   = tid[:max_seq_len]
        ret   = ret[:max_seq_len]
        lat   = lat[:max_seq_len]

    return {
        "call":    np.array(call,  dtype=np.int32),
        "entry":   np.array(entry, dtype=np.int32),
        "dur":     np.array(dur,   dtype=np.int64),
        "proc":    np.array(proc,  dtype=np.int32),
        "pid":     np.array(pid,   dtype=np.int32),
        "tid":     np.array(tid,   dtype=np.int32),
        "ret":     np.array(ret,   dtype=np.int32),
        "lat":     np.array(lat,   dtype=np.int32),
    }


###############################################################################
# Model inference
###############################################################################

_CRIT_E = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
_CRIT_L = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

@torch.no_grad()
def run_inference(model, encoded: dict, device, lat_score_weight: float, model_type: str):
    """
    Run forward pass on a single encoded window.
    Returns (anomaly_score, inference_time_ms).
    """
    def t(arr, dtype=torch.long):
        return torch.from_numpy(arr).unsqueeze(0).to(device, dtype=dtype)  # (1, L)

    call  = t(encoded["call"])
    entry = t(encoded["entry"])
    dur   = t(encoded["dur"])
    proc  = t(encoded["proc"])
    pid   = t(encoded["pid"])
    tid   = t(encoded["tid"])
    ret   = t(encoded["ret"])
    lat   = t(encoded["lat"])

    # Shift: input = [:-1], target = [1:]
    inp_call  = call[:, :-1];  tgt_call  = call[:, 1:]
    inp_entry = entry[:, :-1]; inp_dur   = dur[:, :-1]
    inp_proc  = proc[:, :-1];  inp_pid   = pid[:, :-1]
    inp_tid   = tid[:, :-1];   inp_ret   = ret[:, :-1]
    tgt_lat   = lat[:, 1:]

    t0 = time.perf_counter()

    if model_type == "transformer":
        B, L = inp_call.shape
        idx = torch.arange(L, device=device).unsqueeze(0)
        seq_lens = (inp_call != 0).sum(1)
        pad_mask = idx >= seq_lens.unsqueeze(1)
        logits_e, logits_l = model(inp_call, inp_entry, inp_dur, inp_proc,
                                   inp_pid, inp_tid, inp_ret,
                                   pad_mask=pad_mask, chk=False)
    else:
        logits_e, logits_l = model(inp_call, inp_entry, inp_dur,
                                   inp_proc, inp_pid, inp_tid, inp_ret)

    inf_ms = (time.perf_counter() - t0) * 1000.0

    # Per-sequence mean cross-entropy score (higher = more anomalous)
    B, L, V = logits_e.shape
    per_tok_e = _CRIT_E(logits_e.reshape(B * L, V), tgt_call.reshape(B * L)).reshape(B, L)
    mask_e    = tgt_call != 0
    score_e   = (per_tok_e * mask_e).sum(1) / mask_e.sum(1).clamp(min=1)

    if logits_l.numel() > 0 and lat_score_weight > 0:
        _, _, C = logits_l.shape
        per_tok_l = _CRIT_L(logits_l.reshape(B * L, C), tgt_lat.reshape(B * L)).reshape(B, L)
        mask_l    = tgt_lat != 0
        score_l   = (per_tok_l * mask_l).sum(1) / mask_l.sum(1).clamp(min=1)
        score = ((1.0 - lat_score_weight) * score_e + lat_score_weight * score_l).item()
    else:
        score = score_e.item()

    return score, inf_ms


###############################################################################
# Statistics tracker
###############################################################################

class Stats:
    def __init__(self):
        self.n_windows     = 0
        self.n_dropped     = 0     # async only: windows dropped due to full queue
        self.total_inf_ms  = 0.0
        self.scores        = []
        self._lock         = threading.Lock()
        self._t_start      = time.time()

    def record(self, score: float, inf_ms: float):
        with self._lock:
            self.n_windows    += 1
            self.total_inf_ms += inf_ms
            self.scores.append(score)

    def log_summary(self):
        with self._lock:
            elapsed = time.time() - self._t_start
            n = max(self.n_windows, 1)
            log.info(
                "windows=%d  dropped=%d  avg_inf=%.1f ms  wall=%.0f s  rate=%.1f win/s  "
                "score_mean=%.4f  score_p95=%.4f",
                self.n_windows, self.n_dropped,
                self.total_inf_ms / n, elapsed, self.n_windows / max(elapsed, 1),
                np.mean(self.scores) if self.scores else 0,
                np.percentile(self.scores, 95) if len(self.scores) >= 20 else float("nan"),
            )


###############################################################################
# CSV log writer
###############################################################################

def make_csv_logger(log_file: str):
    """Returns a function that appends one row per window to a CSV file."""
    if log_file is None:
        return lambda **kw: None
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    f = open(log_file, "w", buffering=1)
    f.write("timestamp_iso,n_events,score,inf_ms,queue_depth\n")
    def write_row(*, n_events, score, inf_ms, queue_depth=0):
        f.write(f"{datetime.utcnow().isoformat()},{n_events},{score:.6f},"
                f"{inf_ms:.2f},{queue_depth}\n")
    return write_row


###############################################################################
# Sync mode: single-threaded — inference blocks the collection loop
###############################################################################

def run_sync(args, model, dict_sys, dict_proc, delay_spans, device, stats, log_row):
    segmenter = WindowSegmenter(args.window_ms, args.min_events)

    log.info("MODE: SYNC — inference will block event collection on each window")
    for ev in stream_events_from_bt2(args.trace_dir, args.replay):
        window = segmenter.feed(ev)
        if window is None:
            continue
        events, dur_ms = window
        encoded = encode_window(events, dict_sys, dict_proc, delay_spans,
                                args.n_categories, args.max_seq_len)
        if encoded is None:
            continue
        score, inf_ms = run_inference(model, encoded, device,
                                      args.lat_score_weight, args.model_type)
        stats.record(score, inf_ms)
        log_row(n_events=len(events), score=score, inf_ms=inf_ms)
        if stats.n_windows % 100 == 0:
            stats.log_summary()


###############################################################################
# Async mode: producer/consumer — inference in a separate thread
###############################################################################

def run_async(args, model, dict_sys, dict_proc, delay_spans, device, stats, log_row):
    win_queue = queue.Queue(maxsize=args.queue_maxsize)
    stop_evt  = threading.Event()

    def inference_worker():
        while not stop_evt.is_set() or not win_queue.empty():
            try:
                events, dur_ms = win_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            encoded = encode_window(events, dict_sys, dict_proc, delay_spans,
                                    args.n_categories, args.max_seq_len)
            if encoded is not None:
                score, inf_ms = run_inference(model, encoded, device,
                                              args.lat_score_weight, args.model_type)
                stats.record(score, inf_ms)
                log_row(n_events=len(events), score=score, inf_ms=inf_ms,
                        queue_depth=win_queue.qsize())
                if stats.n_windows % 100 == 0:
                    stats.log_summary()
            win_queue.task_done()

    log.info("MODE: ASYNC — inference in background thread (queue maxsize=%d)", args.queue_maxsize)
    worker = threading.Thread(target=inference_worker, name="inference", daemon=True)
    worker.start()

    segmenter = WindowSegmenter(args.window_ms, args.min_events)
    for ev in stream_events_from_bt2(args.trace_dir, args.replay):
        window = segmenter.feed(ev)
        if window is None:
            continue
        events, dur_ms = window
        try:
            win_queue.put_nowait((events, dur_ms))
        except queue.Full:
            # Drop oldest to make room — don't block the collection side
            try:
                win_queue.get_nowait()
                win_queue.task_done()
            except queue.Empty:
                pass
            with stats._lock:
                stats.n_dropped += 1
            try:
                win_queue.put_nowait((events, dur_ms))
            except queue.Full:
                pass

    stop_evt.set()
    worker.join()


###############################################################################
# Main
###############################################################################

def main():
    args   = get_args()
    device = torch.device("cpu")   # GCP VM has no GPU

    # ── Torch settings optimised for CPU inference ───────────────────────────
    torch.set_num_threads(args.torch_threads)
    torch.set_grad_enabled(False)
    log.info("PyTorch CPU threads: %d", args.torch_threads)

    # ── Load vocab and latency boundaries ───────────────────────────────────
    log.info("Loading vocab from %s", args.vocab_path)
    with open(args.vocab_path, "rb") as f:
        dict_sys, dict_proc = pickle.load(f)
    n_syscall = len(dict_sys)
    n_process = len(dict_proc)
    log.info("Vocab: %d syscalls / %d processes", n_syscall, n_process)

    log.info("Loading latency boundaries from %s", args.delay_spans_path)
    with open(args.delay_spans_path, "rb") as f:
        delay_spans = pickle.load(f)
    # delay_spans may be a dict name→boundaries or a tuple (boundaries_dict, ...)
    if isinstance(delay_spans, tuple):
        delay_spans = delay_spans[0]
    # Normalise any per-entry tuples: (boundaries, other) → boundaries
    delay_spans = {
        k: (v[0] if isinstance(v, tuple) else v)
        for k, v in delay_spans.items()
    }
    log.info("Latency boundaries loaded for %d event types", len(delay_spans))

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_model(args, n_syscall, n_process, device)
    # Warmup to trigger JIT compilation / libtorch caches
    log.info("Warming up model (3 random passes)...")
    _dummy_enc = {
        "call":  np.array([2, 5, 6, 3], dtype=np.int32),
        "entry": np.array([0, 1, 2, 0], dtype=np.int32),
        "dur":   np.array([0, 1000, 500, 0], dtype=np.int64),
        "proc":  np.array([2, 5, 5, 3], dtype=np.int32),
        "pid":   np.zeros(4, dtype=np.int32),
        "tid":   np.array([0, 1, 1, 0], dtype=np.int32),
        "ret":   np.array([0, 0, 1, 0], dtype=np.int32),
        "lat":   np.array([0, 0, 2, 0], dtype=np.int32),
    }
    for _ in range(3):
        run_inference(model, _dummy_enc, device, args.lat_score_weight, args.model_type)
    log.info("Model warmed up.")

    # ── Set up CSV logger ────────────────────────────────────────────────────
    log_row = make_csv_logger(args.log_file)

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = Stats()
    t0 = time.time()

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        if args.mode == "sync":
            run_sync(args, model, dict_sys, dict_proc, delay_spans, device, stats, log_row)
        else:
            run_async(args, model, dict_sys, dict_proc, delay_spans, device, stats, log_row)
    except KeyboardInterrupt:
        log.info("Interrupted.")

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    stats.log_summary()
    log.info(
        "DONE — mode=%s  windows_processed=%d  windows_dropped=%d  "
        "wall_time=%.0f s",
        args.mode, stats.n_windows, stats.n_dropped, elapsed,
    )
    if stats.n_windows > 0:
        log.info(
            "Inference timing — avg=%.1f ms  p50=%.1f ms  p95=%.1f ms  p99=%.1f ms",
            np.mean([stats.total_inf_ms / stats.n_windows]),
            np.percentile([stats.total_inf_ms / stats.n_windows], 50),
            np.percentile([stats.total_inf_ms / stats.n_windows], 95),
            np.percentile([stats.total_inf_ms / stats.n_windows], 99),
        )


if __name__ == "__main__":
    main()
