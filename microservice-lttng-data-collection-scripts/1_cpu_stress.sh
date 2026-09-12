#!/bin/bash
set -e

if ! command -v stress-ng >/dev/null 2>&1; then
  echo "ERROR: stress-ng is not installed or not on PATH. CPU anomaly cannot be injected." >&2
  exit 1
fi

RUN_ID=${1:-run01}
DURATION=${2:-100}
EXPERIMENT_DIR=~/experiments/anomaly_cpu/$RUN_ID
LOAD_USERS=200               # high load
HOST_CPUS=$(nproc)
CPU_WORKERS=${CPU_WORKERS:-$(( HOST_CPUS * 2 ))}
CPU_METHOD=${CPU_METHOD:-matrixprod}
THINK_MIN=${THINK_MIN:-0.1}
THINK_MAX=${THINK_MAX:-0.3}

mkdir -p "$EXPERIMENT_DIR"/{metrics,load_logs}
RUN_LOG="$EXPERIMENT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1
RUN_START_EPOCH=$(date -u +%s)

echo "💥 ULTRA CPU Stress: $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"


echo "⏳ Warmup for Prometheus/service stability (20s)..."
sleep 20

# 1) Tracing
(cd ~ && ./collect_trace.sh anomaly_cpu "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# 2) ULTRA CPU: 12 cores + L3 cache thrashing
stress-ng \
  --cpu "$CPU_WORKERS" \
  --cpu-method "$CPU_METHOD" \
  --cpu-load 100 \
  --timeout "${DURATION}s" \
  --metrics-brief &
STRESS_PID=$!

echo "🔥 12 cores @ 100% + cache thrash (PID $STRESS_PID)"

# 3) High load
python3 ~/load_generator.py \
  --host http://localhost:80 \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min "$THINK_MIN" \
  --think-max "$THINK_MAX" \
  --log-level DEBUG \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$STRESS_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Cleanup
TRACE_DIR=~/traces/anomaly_cpu/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

echo "⏸️  Prometheus flush..."
sleep 10

START_ISO=$(date -u -d "@$((RUN_START_EPOCH-10))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+10))" '+%Y-%m-%dT%H:%M:%SZ')

STEP=10s RATE_WINDOW=1m ./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

# Summary
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
OTEL_SPANS=$(babeltrace "$TRACE_DIR/ust" 2>/dev/null | grep -c "otel.spans" || echo 0)
BUSINESS_SPANS=$(babeltrace "$TRACE_DIR/ust" 2>/dev/null | grep -c -i "service=carts\|service=orders\|service=shipping\|service=queue-master" || echo 0)

cat << EOF

💥 ULTRA CPU COMPLETE: $RUN_ID
📊 Requests: $REQ_COUNT (expect errors)
🔍 Spans: $OTEL_SPANS
📈 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)
EOF
