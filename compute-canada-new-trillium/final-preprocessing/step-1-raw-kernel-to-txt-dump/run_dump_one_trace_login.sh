#!/bin/bash
set -euo pipefail

# Usage:
#   bash compute-canada-new-trillium/final-preprocessing/run_dump_one_trace_login.sh normal run01
#   bash compute-canada-new-trillium/final-preprocessing/run_dump_one_trace_login.sh anomaly_mem run04

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <scenario> <run>" >&2
  echo "Example: $0 normal run01" >&2
  exit 1
fi

SCENARIO="$1"
RUN="$2"

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data
DUMP_ROOT=$SCRATCH/micro-service-trace-data-txt-dump

RUN_DIR="$TRACE_ROOT/$SCENARIO/$RUN"
OUT_DIR="$DUMP_ROOT/$SCENARIO/$RUN"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Mode         : login-node single-trace dump"
echo "Scenario     : $SCENARIO"
echo "Run          : $RUN"
echo "Run dir      : $RUN_DIR"
echo "Output dir   : $OUT_DIR"
echo "============================================================"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run directory not found: $RUN_DIR" >&2
  exit 1
fi

module load python/3.11.5 >/dev/null 2>&1 || true

if ! command -v babeltrace2 >/dev/null 2>&1; then
  echo "ERROR: babeltrace2 is not available in PATH on this login node." >&2
  echo "Try one of the following:" >&2
  echo "  module spider babeltrace" >&2
  echo "  module spider babeltrace2" >&2
  echo "  which babeltrace2" >&2
  exit 1
fi

if [[ -d "$RUN_DIR/kernel" ]]; then
  echo "[KERNEL] Dumping $SCENARIO/$RUN ..."
  time babeltrace2 "$RUN_DIR/kernel" > "$OUT_DIR/kernel.txt"
else
  echo "[WARN] Missing kernel trace: $RUN_DIR/kernel"
fi

if [[ -d "$RUN_DIR/ust" ]]; then
  echo "[UST] Dumping $SCENARIO/$RUN ..."
  time babeltrace2 "$RUN_DIR/ust" > "$OUT_DIR/ust.txt"
else
  echo "[WARN] Missing ust trace: $RUN_DIR/ust"
fi

echo "============================================================"
echo "Done."
echo "Kernel dump : $OUT_DIR/kernel.txt"
echo "UST dump    : $OUT_DIR/ust.txt"
echo "============================================================"
