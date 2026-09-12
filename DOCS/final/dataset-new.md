# SockShop LMAT Dataset: New Kernel-Centric Collection

## Overview

This document describes the new SockShop microservice dataset collected for LMAT using raw LTTng traces on the GCP VM and prepared for training on Compute Canada.

The key difference from the earlier dataset iteration is that this collection now clearly preserves useful per-event metadata in the kernel trace, including:

- `procname`
- `pid`
- `tid`
- syscall entry/exit structure

That makes this dataset much closer to the LMAT paper methodology, where event modeling relies on:

- system call identity
- process name
- process ID
- thread ID
- delay since previous event
- return-status signal
- entry/exit duration recovery

## Scenarios and runs

The dataset contains five scenarios, each with five independent runs:

- `normal`
- `anomaly_cpu`
- `anomaly_disk`
- `anomaly_mem`
- `anomaly_net`

The intended default split for LMAT preprocessing is:

- `train_id`: `normal/run01`, `normal/run02`, `normal/run03`
- `valid_id`: `normal/run04`
- `test_id`: `normal/run05`
- `valid_ood_cpu`: `anomaly_cpu/run04`
- `test_ood_cpu`: `anomaly_cpu/run05`
- `valid_ood_disk`: `anomaly_disk/run04`
- `test_ood_disk`: `anomaly_disk/run05`
- `valid_ood_mem`: `anomaly_mem/run04`
- `test_ood_mem`: `anomaly_mem/run05`
- `valid_ood_net`: `anomaly_net/run04`
- `test_ood_net`: `anomaly_net/run05`

This split matches the current preprocessing pipeline and keeps training vocabulary and duration boundaries learned only from normal runs.

## Directory layout

On the Compute Canada scratch space the raw traces are arranged as:

```text
/scratch/yuvraj17/adaptive_tracing_scratch/micro-service-trace-data/
├── normal/
│   ├── run01/
│   │   ├── kernel/
│   │   ├── ust/
│   │   └── meta/
│   ├── run02/
│   ├── run03/
│   ├── run04/
│   └── run05/
├── anomaly_cpu/
├── anomaly_disk/
├── anomaly_mem/
└── anomaly_net/
```

Each run directory contains:

- `kernel/`: raw LTTng kernel CTF trace
- `ust/`: raw LTTng Python-domain UST events from the OpenTelemetry relay
- `meta/`: sidecar metadata snapshots such as Docker/container/process state

## How the data was collected

The collection pipeline uses:

- LTTng kernel tracing with full kernel event capture
- LTTng userspace tracing for relayed OTel span export events
- a custom SockShop load generator
- anomaly injection scripts for CPU, disk, memory, and network stress
- periodic metadata snapshots for later attribution/debugging

### Kernel tracing

The kernel session is started with full kernel events enabled, which means the raw trace contains:

- syscall events such as `syscall_entry_*` and `syscall_exit_*`
- scheduler events such as `sched_switch`, `sched_wakeup`, `sched_stat_runtime`
- timer events such as `timer_hrtimer_*`
- RCU tracepoints
- power events
- other kernel tracepoints enabled by `-k --all '*'`

In the new traces, kernel events also carry useful context:

- `pid`
- `tid`
- `procname`

This is important because it means the new dataset does not suffer from the earlier issue where `proc` and `pid` were effectively dead features.

### UST tracing

The UST trace comes from a host-side Python relay that reads exported OpenTelemetry spans from container logs and re-emits them as LTTng Python events.

These UST events are useful for cross-layer inspection and documentation, but they are still export-time log events, not clean start/end span pairs. For that reason, the kernel trace remains the primary LMAT training signal.

## What the raw traces look like

### Sample kernel trace

This sample is taken directly from `babeltrace2 kernel` on `normal/run01`:

```text
[00:42:51.993131831] (+?.?????????) lttng-traces-microservice sched_switch: { cpu_id = 11 }, { pid = 50594, tid = 50605, procname = "dockerd" }, { prev_comm = "dockerd", prev_tid = 50605, prev_prio = 20, prev_state = 1, next_comm = "python3", next_tid = 78339, next_prio = 20 }
[00:42:51.993132201] (+0.000000370) lttng-traces-microservice timer_hrtimer_cancel: { cpu_id = 10 }, { pid = 52406, tid = 66036, procname = "java" }, { hrtimer = 0xFFFFCE3B4A0A7570 }
[00:42:51.993135361] (+0.000000340) lttng-traces-microservice syscall_exit_futex: { cpu_id = 11 }, { pid = 78339, tid = 78339, procname = "python3" }, { ret = 0, uaddr = 108935974470848, uaddr2 = 0 }
[00:42:51.993139351] (+0.000000000) lttng-traces-microservice syscall_entry_futex: { cpu_id = 11 }, { pid = 78339, tid = 78339, procname = "python3" }, { uaddr = 108935595519116, op = 137, val = 0, utime = 140721120765888, uaddr2 = 0, val3 = 4294967295 }
[00:42:51.993144411] (+0.000000170) lttng-traces-microservice sched_switch: { cpu_id = 0 }, { pid = 0, tid = 0, procname = "swapper/0" }, { prev_comm = "swapper/0", prev_tid = 0, prev_prio = 20, prev_state = 0, next_comm = "java", next_tid = 78397, next_prio = 20 }
```

This sample confirms:

- syscall entry/exit pairs are present
- `pid`, `tid`, and `procname` are present in the kernel trace
- the trace also includes non-syscall kernel events

### Sample UST trace

This sample is taken directly from `babeltrace2 ust` on `normal/run01`:

```text
[00:42:59.336311000] (+?.?????????) lttng-traces-microservice lttng_python:event: { cpu_id = 6 }, { vpid = 79158, vtid = 79158, procname = "python3" }, { asctime = "2026-03-24 04:42:59,335", msg = "service=carts container=docker-compose_carts_1 phase=export docker_ts=2026-03-24T04:42:59.335401499Z relay_ts_ns=1774327379335924919 trace_id=6424612383807cd1189c15a8d3236908 span_id=2500aa8dabb2415e kind=CLIENT op=\"find data.cart\"", logger_name = "otel.spans", funcName = "<module>", lineno = 86, int_loglevel = 20, thread = 712822784, threadName = "MainThread" }
[00:42:59.336579710] (+0.000095490) lttng-traces-microservice lttng_python:event: { cpu_id = 6 }, { vpid = 79158, vtid = 79158, procname = "python3" }, { asctime = "2026-03-24 04:42:59,336", msg = "service=carts container=docker-compose_carts_1 phase=export docker_ts=2026-03-24T04:42:59.336081669Z relay_ts_ns=1774327379336532249 trace_id=6424612383807cd1189c15a8d3236908 span_id=20265166d8582c69 kind=SERVER op=\"GET /carts/{customerId:.*}/items\"", logger_name = "otel.spans", funcName = "<module>", lineno = 86, int_loglevel = 20, thread = 712822784, threadName = "MainThread" }
```

This confirms that the UST side contains:

- service name
- container name
- trace ID
- span ID
- span kind
- operation name

but still as export events rather than explicit span start/end boundaries.

## How the LMAT dataset is formed

The new preprocessing path is implemented in [preprocess_lmat_kernel.py](/C:/workplace/adaptive_tracer/microservice/preprocess_lmat_kernel.py).

### Segmentation strategy

The current LMAT dataset uses fixed-duration windows:

- segmentation mode: time-window
- default window: `100 ms`
- grouping key: `tid`
- minimum events per window: `8`

This means each thread contributes a sequence of events collected over a rolling 100 ms interval.

### Event scope

The preprocessor supports two scopes:

- `syscall`
- `all`

For paper-aligned LMAT training, `syscall` is the recommended default because the methodology section is explicitly centered on syscall sequence modeling and syscall duration recovery.

Under `syscall` mode:

- only `syscall_entry_*` and `syscall_exit_*` events are kept
- scheduler/timer/RCU/power events are ignored
- durations are recovered by pairing syscall entry and exit events for the same `(event_name, tid)` context

Under `all` mode:

- all kernel tracepoints are included
- only entry/exit syscall events contribute real latency targets
- non-syscall events remain useful as contextual sequence tokens

### Event representation

Each event is encoded using the LMAT-style feature channels:

- `call`: event vocabulary index
- `entry`: `0/1/2` for neither/entry/exit
- `duration`: delay since previous event
- `proc`: process vocabulary index
- `pid`
- `tid`
- `ret`: return-status encoding
- `lat_cat`: duration category for exit events

Special tokens are used for:

- start of sequence
- end of sequence
- truncation

### Duration modeling

The dataset supports the paper’s duration modeling setup:

- match syscall entry and exit events
- compute execution duration in nanoseconds
- learn duration bin boundaries from training normal data only
- apply those fixed boundaries to valid/test/OOD splits

This makes the resulting dataset compatible with:

- event-only training
- duration-only training
- multi-task training

using the same shards.

## Output shard format

The preprocessing output is written as NPZ shards for the current training pipeline.

At the root of the preprocessed directory:

- `vocab.pkl`
- `delay_spans.pkl`
- `dataset_manifest.json`

For each split:

- `shard_*.npz`
- `meta.json`

Each shard contains:

- `call`
- `entry`
- `duration`
- `proc`
- `pid`
- `tid`
- `ret`
- `lat_cat`
- `seq_len`
- `req_dur_ms`
- `is_anomaly`

## Why this dataset is stronger than the earlier version

Compared with the earlier SockShop dataset iteration, this collection is stronger for LMAT because:

- kernel `procname` is present
- kernel `pid` is present
- syscall entry/exit pairs are clearly recoverable
- the raw traces align more directly with the paper’s LMAT feature design

This means the new dataset is much better suited for studying the real contribution of:

- event modeling
- duration modeling
- multi-task learning

without the earlier ambiguity caused by dead process metadata.

## Recommended preprocessing configuration

For the paper-aligned dataset, the recommended settings are:

- `event_scope=syscall`
- `window_ms=100`
- `warmup_s=5`
- `min_events=8`
- `max_seq_len=512`
- `n_categories=6`

This configuration produces a clean LMAT dataset focused on syscall behavior while preserving the feature channels required for event-only, duration-only, and multi-task experiments.

## Recommended documentation statement

Suggested paper/report wording:

> The new SockShop LMAT dataset is constructed directly from raw LTTng kernel traces collected under five operating scenarios: normal, CPU stress, disk stress, memory stress, and network stress. Each scenario contains five independent runs. Unlike the earlier dataset iteration, the new kernel traces preserve meaningful per-event `procname`, `pid`, and `tid` fields, enabling event representations that closely follow the LMAT methodology described in the paper. Training shards are built from fixed 100 ms TID-based windows, and syscall execution durations are recovered by matching `syscall_entry_*` and `syscall_exit_*` events within the same thread context. Duration bins are learned from normal training runs only and then reused across validation, test, and OOD splits.

