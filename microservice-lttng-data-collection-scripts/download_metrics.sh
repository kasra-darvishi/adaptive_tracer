#!/usr/bin/env bash
set -u
set -o pipefail

PROMETHEUS="${PROMETHEUS:-http://136.114.233.241:9090}"
STEP="${STEP:-10s}"
RATE_WINDOW="${RATE_WINDOW:-1m}"

usage() {
    cat <<EOF
Usage: $0 <start> <end> [output_dir]
EOF
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "❌ Required command not found: $1"
        exit 1
    }
}

log() {
    echo "$@"
}

require_cmd date
require_cmd curl
require_cmd wc
require_cmd du
require_cmd ls
require_cmd mktemp

HAS_JQ=0
if command -v jq >/dev/null 2>&1; then
    HAS_JQ=1
fi

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    usage
fi

START_RAW=$1
END_RAW=$2
OUTPUT_DIR=${3:-metrics}

START=$(date -u -d "$START_RAW" +%s 2>/dev/null) || {
    echo "❌ Invalid start time: $START_RAW"
    exit 1
}

END=$(date -u -d "$END_RAW" +%s 2>/dev/null) || {
    echo "❌ Invalid end time: $END_RAW"
    exit 1
}

if [ "$START" -ge "$END" ]; then
    echo "❌ Start time must be earlier than end time"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" || {
    echo "❌ Could not create output directory: $OUTPUT_DIR"
    exit 1
}

log "📥 Downloading: $START_RAW → $END_RAW"
log "📁 Output dir: $OUTPUT_DIR/"
log "⏱️  Range UTC: $(date -u -d "@$START" '+%Y-%m-%d %H:%M:%S UTC') → $(date -u -d "@$END" '+%Y-%m-%d %H:%M:%S UTC')"
log "🌐 Prometheus: $PROMETHEUS"
log "📏 Step: $STEP"
log "🪟 Rate window: $RATE_WINDOW"
echo

log "=== Connectivity Test ==="
HEALTH_TMP=$(mktemp)
HTTP_CODE=$(curl -sS -m 15 -w "%{http_code}" -o "$HEALTH_TMP" \
    "$PROMETHEUS/api/v1/query?query=up")

if [ "$HTTP_CODE" != "200" ]; then
    echo "❌ Could not reach Prometheus. HTTP $HTTP_CODE"
    cat "$HEALTH_TMP"
    rm -f "$HEALTH_TMP"
    exit 1
fi

if [ "$HAS_JQ" -eq 1 ]; then
    STATUS=$(jq -r '.status // "unknown"' "$HEALTH_TMP" 2>/dev/null)
    if [ "$STATUS" != "success" ]; then
        echo "❌ Prometheus responded, but query failed:"
        cat "$HEALTH_TMP"
        rm -f "$HEALTH_TMP"
        exit 1
    fi
fi

rm -f "$HEALTH_TMP"
log "✅ Prometheus reachable"
echo

download() {
    local name="$1"
    local query="$2"
    local file="$OUTPUT_DIR/$name.json"
    local tmp
    tmp=$(mktemp)

    printf "→ %-24s " "$name"

    local http_code
    http_code=$(curl -sS -m 60 -w "%{http_code}" -o "$tmp" -G \
        "$PROMETHEUS/api/v1/query_range" \
        --data-urlencode "query=$query" \
        --data-urlencode "start=$START" \
        --data-urlencode "end=$END" \
        --data-urlencode "step=$STEP")

    if [ "$http_code" != "200" ]; then
        mv "$tmp" "$file"
        echo "❌ HTTP $http_code"
        return 1
    fi

    mv "$tmp" "$file"

    if [ "$HAS_JQ" -eq 1 ]; then
        local status result_count err_type err_msg
        status=$(jq -r '.status // "unknown"' "$file" 2>/dev/null)
        result_count=$(jq -r '(.data.result | length) // 0' "$file" 2>/dev/null)
        err_type=$(jq -r '.errorType // empty' "$file" 2>/dev/null)
        err_msg=$(jq -r '.error // empty' "$file" 2>/dev/null)

        if [ "$status" = "error" ]; then
            echo "❌ Prometheus error: ${err_type:-unknown} ${err_msg:-}"
            return 1
        elif [ "$status" != "success" ]; then
            echo "❌ Unknown response"
            return 1
        elif [ "$result_count" -eq 0 ]; then
            echo "⚠️  no data"
            return 0
        else
            local size
            size=$(wc -c < "$file")
            echo "✅ $result_count series, ${size} bytes"
            return 0
        fi
    else
        local size
        size=$(wc -c < "$file")
        echo "✅ saved (${size} bytes)"
        return 0
    fi
}

FAILED=0

echo "=== VM Metrics ==="

download "vm_cpu" \
"100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[$RATE_WINDOW])) * 100)" || FAILED=1

download "vm_memory" \
"(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100" || FAILED=1

download "vm_disk" \
"100 - (
  node_filesystem_avail_bytes{fstype!=\"rootfs\",mountpoint=\"/\"}
  /
  node_filesystem_size_bytes{fstype!=\"rootfs\",mountpoint=\"/\"}
) * 100" || FAILED=1

download "vm_network_receive" \
"rate(node_network_receive_bytes_total{device!=\"lo\"}[$RATE_WINDOW])" || FAILED=1

download "vm_network_transmit" \
"rate(node_network_transmit_bytes_total{device!=\"lo\"}[$RATE_WINDOW])" || FAILED=1

echo

SERVICES="catalogue cart orders payment shipping user frontend"

echo "=== Sock Shop Services ==="
for service in $SERVICES; do
    echo "--- $service ---"

    download "${service}_qps" \
"sum(rate(request_duration_seconds_count{job=\"$service\"}[$RATE_WINDOW]))" || FAILED=1

    download "${service}_p95_latency" \
"histogram_quantile(
  0.95,
  sum(rate(request_duration_seconds_bucket{job=\"$service\"}[$RATE_WINDOW])) by (le)
)" || FAILED=1

    download "${service}_p50_latency" \
"histogram_quantile(
  0.50,
  sum(rate(request_duration_seconds_bucket{job=\"$service\"}[$RATE_WINDOW])) by (le)
)" || FAILED=1

    download "${service}_errors" \
"sum(rate(request_duration_seconds_count{job=\"$service\",status_code=~\"5..\"}[$RATE_WINDOW]))" || FAILED=1

    echo
done

FILE_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')
DIR_SIZE=$(du -sh "$OUTPUT_DIR" | awk '{print $1}')

echo "🎉 Complete"
echo "📁 Files written: $FILE_COUNT"
echo "💾 Directory size: $DIR_SIZE"
echo

ls -lh "$OUTPUT_DIR" | head -20

echo
if [ "$FAILED" -eq 1 ]; then
    echo "⚠️  Finished with some failed queries. Check the corresponding JSON files for details."
    exit 2
else
    echo "✅ All queries completed."
    exit 0
fi
