#!/bin/bash
# baseline_load.sh — Pure baseline: NO LTTng, NO OTel relay, NO LMAT
# Provides the instrumentation-free floor for overhead comparisons.
#
# Usage: ./baseline_load.sh <run_id> [duration_seconds]
#   e.g. ./baseline_load.sh run01 300
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ID=${1:-run01}
DURATION=${2:-300}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-~/experiments}
EXPERIMENT_DIR=$EXPERIMENT_ROOT/baseline/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}
THINK_MIN=${THINK_MIN:-0.2}
THINK_MAX=${THINK_MAX:-1.0}
LOAD_GENERATOR=${LOAD_GENERATOR:-$SCRIPT_DIR/load_generator.py}
WARMUP_DURATION=${WARMUP_DURATION:-0}

mkdir -p "$EXPERIMENT_DIR"/load_logs
RUN_LOG="$EXPERIMENT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "🚀 BASELINE (no tracing, no LMAT): $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"
echo "   Host=$FRONTEND_HOST  think=${THINK_MIN}-${THINK_MAX}s  root=$EXPERIMENT_ROOT  warmup=${WARMUP_DURATION}s"

# ── Stop ALL tracing (idempotent — safe even if no sessions exist) ──────────
echo "🔇 Destroying any active LTTng sessions..."
lttng destroy --all 2>/dev/null || true
sudo lttng destroy --all 2>/dev/null || true

# ── Kill OTel relay if running ───────────────────────────────────────────────
echo "🔇 Killing OTel relay (if running)..."
pkill -f otel-to-lttng.py 2>/dev/null || true
sleep 1   # give it a moment to die

# ── Confirm nothing is tracing ───────────────────────────────────────────────
ACTIVE=$(lttng list 2>/dev/null | grep -c "Recording session" || true)
SUDO_ACTIVE=$(sudo lttng list 2>/dev/null | grep -c "Recording session" || true)
if [[ "$ACTIVE" -gt 0 ]] || [[ "$SUDO_ACTIVE" -gt 0 ]]; then
    echo "⚠️  WARNING: LTTng sessions still active. Aborting." >&2
    exit 1
fi
echo "✅ LTTng is silent. Starting pure baseline run."

if [[ "$WARMUP_DURATION" -gt 0 ]]; then
    echo "🔥 Warm-up load for ${WARMUP_DURATION}s before measured baseline run ..."
    python3 "$LOAD_GENERATOR" \
        --host "$FRONTEND_HOST" \
        --users "$LOAD_USERS" \
        --duration "$WARMUP_DURATION" \
        --think-min "$THINK_MIN" \
        --think-max "$THINK_MAX" \
        --log-level WARNING \
        --output "$EXPERIMENT_DIR/warmup_load_results.csv" >/dev/null 2>&1 || true
    sleep 5
fi

RUN_START_EPOCH=$(date -u +%s)

# ── Load generator only ──────────────────────────────────────────────────────
python3 "$LOAD_GENERATOR" \
    --host "$FRONTEND_HOST" \
    --users "$LOAD_USERS" \
    --duration "$DURATION" \
    --think-min "$THINK_MIN" \
    --think-max "$THINK_MAX" \
    --log-level WARNING \
    --output "$EXPERIMENT_DIR/load_results.csv"

RUN_END_EPOCH=$(date -u +%s)

# ── Summary ──────────────────────────────────────────────────────────────────
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
ELAPSED=$((RUN_END_EPOCH - RUN_START_EPOCH))

cat <<EOF

✅ BASELINE $RUN_ID COMPLETE
📊 Requests    : $REQ_COUNT  (in ${ELAPSED}s)
📈 Throughput  : $(echo "scale=1; $REQ_COUNT / $ELAPSED" | bc) req/s (approx)
📁 Output      : $EXPERIMENT_DIR

EOF
