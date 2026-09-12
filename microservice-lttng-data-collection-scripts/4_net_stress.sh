#!/bin/bash
set -euo pipefail

RUN_ID=${1:-run01}
DURATION=${2:-100}
EXPERIMENT_DIR=~/experiments/net_stress/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}
TRACE_START_DELAY=${TRACE_START_DELAY:-8}
THINK_MIN=${THINK_MIN:-0.1}
THINK_MAX=${THINK_MAX:-0.3}

# More stable impairment profile
NET_DELAY_MS=${NET_DELAY_MS:-80}
NET_JITTER_MS=${NET_JITTER_MS:-20}
NET_LOSS_PCT=${NET_LOSS_PCT:-0.5}
NET_RATE=${NET_RATE:-20mbit}
NET_BURST=${NET_BURST:-64k}
NET_LATENCY=${NET_LATENCY:-100ms}

mkdir -p "$EXPERIMENT_DIR"/{metrics,load_logs}
TRACE_LOG="$EXPERIMENT_DIR/collect_trace.log"
RUN_LOG="$EXPERIMENT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

resolve_sockshop_bridge() {
  local bridge
  bridge=$(docker network inspect -f '{{printf "%.12s" .Id}}' docker-compose_default 2>/dev/null || true)
  if [[ -n "$bridge" ]]; then
    echo "br-$bridge"
    return 0
  fi

  docker network ls --format '{{.Name}}' | grep -qx 'docker-compose_default' || return 1
  return 1
}

# Docker bridge for Sock Shop traffic
NET_IFACE=${NET_IFACE:-$(resolve_sockshop_bridge)}

TRACE_PID=""
LOAD_PID=""

echo "🌐 NET Anomaly: $RUN_ID (${DURATION}s, ${LOAD_USERS} users) iface=${NET_IFACE} delay=${NET_DELAY_MS}ms±${NET_JITTER_MS}ms loss=${NET_LOSS_PCT}% rate=${NET_RATE}"

sudo -v
echo "⏳ Warmup for Prometheus/service stability (20s)..."
sleep 20

RUN_START_EPOCH=$(date -u +%s)

cleanup() {
  sudo tc qdisc del dev "$NET_IFACE" root 2>/dev/null || true

  if [[ -n "${LOAD_PID:-}" ]]; then
    kill "$LOAD_PID" 2>/dev/null || true
    wait "$LOAD_PID" 2>/dev/null || true
  fi

  if [[ -n "${TRACE_PID:-}" ]]; then
    kill "$TRACE_PID" 2>/dev/null || true
    wait "$TRACE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -z "$NET_IFACE" ]]; then
  echo "ERROR: could not resolve the docker-compose_default bridge interface" >&2
  echo "Set NET_IFACE manually, for example: NET_IFACE=br-3b32b24f077e ./net_stress.sh $RUN_ID $DURATION" >&2
  exit 1
fi

if ! ip link show "$NET_IFACE" >/dev/null 2>&1; then
  echo "ERROR: resolved NET_IFACE '$NET_IFACE' does not exist on this host" >&2
  exit 1
fi

echo "Tracing will start first; waiting ${TRACE_START_DELAY}s before applying netem."

# If a previous run was interrupted, stale sessions can block the next one.
lttng destroy sockshop-ust 2>/dev/null || true
sudo lttng destroy sockshop-kernel 2>/dev/null || true

# Tracing
(cd ~ && ./collect_trace.sh anomaly_net "$RUN_ID" "$DURATION") >"$TRACE_LOG" 2>&1 &
TRACE_PID=$!
sleep "$TRACE_START_DELAY"

# Apply network impairment
sudo tc qdisc add dev "$NET_IFACE" root handle 1: netem \
  delay "${NET_DELAY_MS}ms" "${NET_JITTER_MS}ms" distribution normal \
  loss "${NET_LOSS_PCT}%"

sudo tc qdisc add dev "$NET_IFACE" parent 1: handle 10: tbf \
  rate "$NET_RATE" burst "$NET_BURST" latency "$NET_LATENCY"

echo "⚠️  tc netem/tbf applied on $NET_IFACE"
sudo tc qdisc show dev "$NET_IFACE" || true

# Load
python3 ~/load_generator.py \
  --host "$FRONTEND_HOST" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min "$THINK_MIN" \
  --think-max "$THINK_MAX" \
  --log-level DEBUG \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

TRACE_DIR=~/traces/anomaly_net/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

echo "⏸️  Prometheus flush (10s)..."
sleep 10

START_ISO=$(date -u -d "@$((RUN_START_EPOCH-10))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+10))" '+%Y-%m-%dT%H:%M:%SZ')

STEP=10s RATE_WINDOW=1m ./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
OTEL_SPANS=$(babeltrace "$TRACE_DIR/ust" 2>/dev/null | grep -c "otel.spans" || echo 0)
BUSINESS_SPANS=$(babeltrace "$TRACE_DIR/ust" 2>/dev/null | grep -c -i "service=carts\|service=orders\|service=shipping\|service=queue-master" || echo 0)

cat << EOF

✅ $RUN_ID COMPLETE
📊 Requests: $REQ_COUNT
🔍 Spans: $OTEL_SPANS ($BUSINESS_SPANS business)
📈 Metrics: $(find "$EXPERIMENT_DIR/metrics" -type f | wc -l) files
💾 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)
💾 $(du -sh "$TRACE_DIR" 2>/dev/null | cut -f1)

EOF
