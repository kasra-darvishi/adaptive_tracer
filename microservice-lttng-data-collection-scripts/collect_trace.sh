#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TYPE=$1
RUN=$2
DURATION=${3:-120}
EXTRA_ARG=${4:-}
OUTPUT_DIR=~/traces/$TYPE/$RUN
META_DIR="$OUTPUT_DIR/meta"

mkdir -p "$OUTPUT_DIR"/{kernel,ust}
mkdir -p "$META_DIR"

snapshot_metadata() {
    local tag="$1"

    {
        echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "hostname=$(hostname)"
        echo "kernel=$(uname -r)"
        echo "type=$TYPE"
        echo "run=$RUN"
        echo "duration=$DURATION"
    } > "$META_DIR/runinfo_${tag}.txt"

    docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' \
        > "$META_DIR/docker_ps_${tag}.txt" 2>&1 || true

    ps -eLo pid,tid,ppid,psr,cls,pri,stat,comm,cmd \
        > "$META_DIR/ps_threads_${tag}.txt" 2>&1 || true

    mapfile -t SOCKSHOP_CONTAINERS < <(
        docker ps --format '{{.Names}}' | grep -E 'docker-compose_(front-end|edge-router|carts|orders|payment|shipping|user|catalogue|catalogue-db|carts-db|orders-db|user-db|session-db|queue-master|rabbitmq)_1' || true
    )

    printf "%s\n" "${SOCKSHOP_CONTAINERS[@]}" \
        > "$META_DIR/container_list_${tag}.txt"

    for c in "${SOCKSHOP_CONTAINERS[@]}"; do
        safe_name="${c//\//_}"

        docker inspect "$c" > "$META_DIR/inspect_${safe_name}_${tag}.json" 2>&1 || true
        docker top "$c" -eo pid,tid,ppid,psr,stat,comm,args > "$META_DIR/top_${safe_name}_${tag}.txt" 2>&1 || true

        pid="$(docker inspect -f '{{.State.Pid}}' "$c" 2>/dev/null || echo '')"
        if [[ -n "$pid" && "$pid" != "0" ]]; then
            {
                echo "container=$c"
                echo "host_pid=$pid"
                echo
                cat "/proc/$pid/cgroup" 2>/dev/null || true
                echo
                ls -l "/proc/$pid/ns" 2>/dev/null || true
            } > "$META_DIR/proc_${safe_name}_${tag}.txt"
        fi
    done
}

lttng create sockshop-ust --output="$OUTPUT_DIR/ust"
lttng enable-event --python otel.spans
lttng add-context --userspace --type=vpid --type=vtid --type=procname 2>/dev/null || true
lttng start

sudo lttng create sockshop-kernel --output="$OUTPUT_DIR/kernel"
sudo lttng enable-event -k --all '*'
sudo lttng add-context --kernel --type=pid --type=tid --type=procname 2>/dev/null || true
sudo lttng start

snapshot_metadata "start"

(
    while true; do
        sleep 10
        snapshot_metadata "tick_$(date -u +%Y%m%dT%H%M%SZ)"
    done
) &
SNAP_PID=$!

python3 "$SCRIPT_DIR/agents/otel-to-lttng.py" ${EXTRA_ARG:+$EXTRA_ARG} &
RELAY_PID=$!

echo "[$TYPE/$RUN] FULL TRACING (Kernel+UST) for ${DURATION}s..."
sleep "$DURATION"

kill "$SNAP_PID" 2>/dev/null || true
wait "$SNAP_PID" 2>/dev/null || true

kill "$RELAY_PID" 2>/dev/null || true
wait "$RELAY_PID" 2>/dev/null || true

snapshot_metadata "end"

lttng stop || true
sudo lttng stop || true
lttng destroy || true
sudo lttng destroy || true

echo "$TYPE,$RUN,$(date -u +%Y-%m-%dT%H:%M:%SZ),${DURATION}s,FULL" >> "$META_DIR/trace_runs.csv"

echo "[$TYPE/$RUN] DONE. $(sudo du -sh "$OUTPUT_DIR")"
