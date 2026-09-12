#!/bin/bash
# lmat_sync_run.sh — LTTng tracing ON + LMAT co-located inference
#
# Two-phase approach (avoids lttng flush requirement):
#   Phase 1: LTTng + load generator run concurrently (captures load_results.csv)
#   Phase 2: online_inference.py --replay processes the completed trace in sync
#            mode while a second load generator run is active, so SockShop
#            experiences the CPU pressure from co-located LMAT inference.
#            The phase-2 load_results.csv is the measurement for the paper.
#
# Usage: ./lmat_sync_run.sh <run_id> [duration_seconds] [--quiet]
#   e.g. ./lmat_sync_run.sh run01 300 --quiet
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
RUN_ID=${1:-run01}
DURATION=${2:-300}
QUIET_FLAG=${3:-}    # pass --quiet to silence the OTel span printing
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-~/experiments}
EXPERIMENT_DIR=$EXPERIMENT_ROOT/lmat_sync/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}
THINK_MIN=${THINK_MIN:-0.2}
THINK_MAX=${THINK_MAX:-1.0}
TORCH_THREADS=${TORCH_THREADS:-2}
LOAD_GENERATOR=${LOAD_GENERATOR:-$SCRIPT_DIR/load_generator.py}

# ── Edit these paths before running on the GCP VM ────────────────────────────
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/logs/lstm_multitask_cats5_seq512_382061/model_best.pt}
VOCAB_PATH=${VOCAB_PATH:-$PROJECT_DIR/micro-service-trace-data/preprocessed_seq512/preprocessed_lmat_kernel_cats5_seq512/vocab.pkl}
DELAY_PATH=${DELAY_PATH:-$PROJECT_DIR/micro-service-trace-data/preprocessed_seq512/preprocessed_lmat_kernel_cats5_seq512/delay_spans.pkl}
MODEL_TYPE=${MODEL_TYPE:-lstm}
N_HIDDEN=${N_HIDDEN:-1024}; N_LAYER=${N_LAYER:-6}; N_HEAD=${N_HEAD:-8}
DIM_SYS=${DIM_SYS:-48};     DIM_ENTRY=${DIM_ENTRY:-12}; DIM_RET=${DIM_RET:-12}
DIM_PROC=${DIM_PROC:-48};   DIM_PID=${DIM_PID:-12};     DIM_TID=${DIM_TID:-12}
DIM_ORDER=${DIM_ORDER:-12}; DIM_TIME=${DIM_TIME:-12}
N_CATEGORIES=${N_CATEGORIES:-6}
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$EXPERIMENT_DIR"

# Fix traces dir permissions (may be root-owned from previous sudo lttng run)
TRACE_DIR=~/traces/lmat_sync/$RUN_ID
sudo mkdir -p "$TRACE_DIR"/{kernel,ust} 2>/dev/null || true
sudo chown -R "$(whoami)" ~/traces/lmat_sync 2>/dev/null || true

echo "🚀 LMAT SYNC: $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"
echo "   Host=$FRONTEND_HOST  think=${THINK_MIN}-${THINK_MAX}s  root=$EXPERIMENT_ROOT  torch_threads=$TORCH_THREADS"

# ── PHASE 1: Collect trace + load data (no inference yet) ───────────────────
echo "📡 Phase 1: LTTng tracing + load generator ($DURATION s)..."
RUN_START_EPOCH=$(date -u +%s)

# LTTng collection
("$SCRIPT_DIR/collect_trace.sh" lmat_sync "$RUN_ID" "$DURATION" $QUIET_FLAG) &
TRACE_PID=$!

# Load generator (this is the primary latency measurement)
python3 "$LOAD_GENERATOR" \
    --host "$FRONTEND_HOST" \
    --users "$LOAD_USERS" \
    --duration "$DURATION" \
    --think-min "$THINK_MIN" \
    --think-max "$THINK_MAX" \
    --log-level WARNING \
    --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$LOAD_PID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

RUN_END_EPOCH=$(date -u +%s)

# ── PHASE 2: Replay inference + concurrent load (CPU interference measurement)
echo "🧠 Phase 2: Replaying trace through model + concurrent load ($DURATION s)..."

# Fresh load run concurrent with inference so SockShop feels the CPU impact
python3 "$LOAD_GENERATOR" \
    --host "$FRONTEND_HOST" \
    --users "$LOAD_USERS" \
    --duration "$DURATION" \
    --think-min "$THINK_MIN" \
    --think-max "$THINK_MAX" \
    --log-level WARNING \
    --output "$EXPERIMENT_DIR/load_results_with_inference.csv" &
LOAD2_PID=$!

# Replay the completed trace through the model in sync mode
python3 "$SCRIPT_DIR/online_inference.py" \
    --model_path       "$MODEL_PATH" \
    --vocab_path       "$VOCAB_PATH" \
    --delay_spans_path "$DELAY_PATH" \
    --trace_dir        "$TRACE_DIR/kernel" \
    --model_type       "$MODEL_TYPE" \
    --n_categories     "$N_CATEGORIES" \
    --n_hidden "$N_HIDDEN" --n_layer "$N_LAYER" --n_head "$N_HEAD" \
    --dim_sys "$DIM_SYS" --dim_entry "$DIM_ENTRY" --dim_ret "$DIM_RET" \
    --dim_proc "$DIM_PROC" --dim_pid "$DIM_PID" --dim_tid "$DIM_TID" \
    --dim_order "$DIM_ORDER" --dim_time "$DIM_TIME" \
    --mode sync \
    --replay \
    --torch_threads "$TORCH_THREADS" \
    --window_ms 100 \
    --log_file "$EXPERIMENT_DIR/inference.log" &
INFER_PID=$!

# Wait for whichever finishes first; kill the other
wait "$INFER_PID" || true
kill "$LOAD2_PID" 2>/dev/null && wait "$LOAD2_PID" 2>/dev/null || true

PHASE2_END=$(date -u +%s)
ELAPSED=$((RUN_END_EPOCH - RUN_START_EPOCH))
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
REQ2_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results_with_inference.csv" 2>/dev/null | wc -l || echo 0)

cat <<EOF

✅ LMAT SYNC $RUN_ID COMPLETE
─────────────────────────────────────────────
Phase 1 (LTTng only):
  📊 Requests   : $REQ_COUNT  (in ${ELAPSED}s)
  📈 Throughput : $(echo "scale=1; $REQ_COUNT / $ELAPSED" | bc) req/s

Phase 2 (with co-located LMAT inference replaying):
  📊 Requests   : $REQ2_COUNT
  🧠 Inference  : $EXPERIMENT_DIR/inference.log

📁 Output dir  : $EXPERIMENT_DIR
─────────────────────────────────────────────
Use load_results.csv for LTTng overhead.
Use load_results_with_inference.csv for LMAT co-located overhead.
EOF
