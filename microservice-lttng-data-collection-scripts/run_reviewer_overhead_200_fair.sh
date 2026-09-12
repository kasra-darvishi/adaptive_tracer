#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-~/experiments/reviewer_overhead_200_fair}
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
USERS=${USERS:-200}
REPEATS=${REPEATS:-3}
DURATION=${DURATION:-100}
WARMUP_DURATION=${WARMUP_DURATION:-30}
THINK_MIN=${THINK_MIN:-0.2}
THINK_MAX=${THINK_MAX:-1.0}
COOLDOWN_S=${COOLDOWN_S:-20}
QUIET_FLAG=${QUIET_FLAG:---quiet}
TORCH_THREADS=${TORCH_THREADS:-2}

MODEL_PATH=${MODEL_PATH:-}
VOCAB_PATH=${VOCAB_PATH:-}
DELAY_PATH=${DELAY_PATH:-}
MODEL_TYPE=${MODEL_TYPE:-lstm}
N_HIDDEN=${N_HIDDEN:-1024}
N_LAYER=${N_LAYER:-6}
N_HEAD=${N_HEAD:-8}
DIM_SYS=${DIM_SYS:-48}
DIM_ENTRY=${DIM_ENTRY:-12}
DIM_RET=${DIM_RET:-12}
DIM_PROC=${DIM_PROC:-48}
DIM_PID=${DIM_PID:-12}
DIM_TID=${DIM_TID:-12}
DIM_ORDER=${DIM_ORDER:-12}
DIM_TIME=${DIM_TIME:-12}
N_CATEGORIES=${N_CATEGORIES:-6}

mkdir -p "$EXPERIMENT_ROOT"
MANIFEST="$EXPERIMENT_ROOT/run_manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
    echo "condition,users,repeat,run_id,duration_s,warmup_s,frontend_host,status,started_utc,ended_utc" > "$MANIFEST"
fi

echo "============================================================"
echo "Reviewer Overhead Fair 200-User Run"
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Host           : $FRONTEND_HOST"
echo "Users          : $USERS"
echo "Repeats        : $REPEATS"
echo "Warmup         : ${WARMUP_DURATION}s"
echo "Duration       : ${DURATION}s"
echo "Think time     : ${THINK_MIN}-${THINK_MAX}s"
echo "Torch threads  : $TORCH_THREADS"
echo "============================================================"

run_condition() {
    local condition="$1"
    local repeat="$2"
    local run_id
    local started ended

    run_id=$(printf "u%03d_r%02d" "$USERS" "$repeat")
    started=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    export EXPERIMENT_ROOT FRONTEND_HOST THINK_MIN THINK_MAX LOAD_USERS TORCH_THREADS WARMUP_DURATION
    export MODEL_PATH VOCAB_PATH DELAY_PATH MODEL_TYPE
    export N_HIDDEN N_LAYER N_HEAD DIM_SYS DIM_ENTRY DIM_RET DIM_PROC DIM_PID DIM_TID DIM_ORDER DIM_TIME N_CATEGORIES
    LOAD_USERS="$USERS"

    echo
    echo ">>> repeat=$repeat condition=$condition run_id=$run_id"

    case "$condition" in
        baseline)
            "$SCRIPT_DIR/baseline_load.sh" "$run_id" "$DURATION"
            ;;
        lttng_only)
            "$SCRIPT_DIR/lttng_only_run.sh" "$run_id" "$DURATION" "$QUIET_FLAG"
            ;;
        lmat_async)
            "$SCRIPT_DIR/lmat_async_run.sh" "$run_id" "$DURATION" "$QUIET_FLAG"
            ;;
        *)
            echo "Unknown condition: $condition" >&2
            return 1
            ;;
    esac

    ended=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "$condition,$USERS,$repeat,$run_id,$DURATION,$WARMUP_DURATION,$FRONTEND_HOST,ok,$started,$ended" >> "$MANIFEST"
}

for repeat in $(seq 1 "$REPEATS"); do
    case "$repeat" in
        1) order="baseline lttng_only lmat_async" ;;
        2) order="lmat_async baseline lttng_only" ;;
        3) order="lttng_only lmat_async baseline" ;;
        *) order="baseline lttng_only lmat_async" ;;
    esac

    echo
    echo "=== Repeat $repeat order: $order ==="
    for condition in $order; do
        run_condition "$condition" "$repeat"
        if [[ "$COOLDOWN_S" -gt 0 ]]; then
            echo "Cooling down for ${COOLDOWN_S}s ..."
            sleep "$COOLDOWN_S"
        fi
    done
done

echo
echo "Fair 200-user run complete."
echo "Manifest: $MANIFEST"
echo
echo "Analyze with:"
echo "python3 \"$SCRIPT_DIR/analyse_reviewer_overhead.py\" --experiment_root \"$EXPERIMENT_ROOT\" --reference_users 200 --max_error_rate_pct 3.0"
