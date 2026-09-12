#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-~/experiments/reviewer_overhead}
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS_SERIES=${LOAD_USERS_SERIES:-"50 100 150 200 250 300"}
REPEATS=${REPEATS:-3}
DURATION=${DURATION:-300}
THINK_MIN=${THINK_MIN:-0.2}
THINK_MAX=${THINK_MAX:-1.0}
COOLDOWN_S=${COOLDOWN_S:-20}
CONDITIONS=${CONDITIONS:-"baseline lttng_only lmat_async"}
QUIET_FLAG=${QUIET_FLAG:---quiet}
TORCH_THREADS=${TORCH_THREADS:-2}

MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/logs/lstm_multitask_cats5_seq512_382061/model_best.pt}
VOCAB_PATH=${VOCAB_PATH:-$PROJECT_DIR/micro-service-trace-data/preprocessed_seq512/preprocessed_lmat_kernel_cats5_seq512/vocab.pkl}
DELAY_PATH=${DELAY_PATH:-$PROJECT_DIR/micro-service-trace-data/preprocessed_seq512/preprocessed_lmat_kernel_cats5_seq512/delay_spans.pkl}
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

usage() {
    cat <<EOF
Usage: $0

Environment overrides:
  EXPERIMENT_ROOT   Root directory for all reviewer-overhead runs
  FRONTEND_HOST     Sock Shop frontend URL
  LOAD_USERS_SERIES Space-separated user counts, e.g. "50 100 150 200"
  REPEATS           Number of repeats per user level
  DURATION          Duration per run in seconds
  THINK_MIN/MAX     Shared think-time across all conditions
  CONDITIONS        Subset of: "baseline lttng_only lmat_async lmat_sync"
  COOLDOWN_S        Sleep between runs to reduce cross-run interference
  TORCH_THREADS     PyTorch CPU threads for LMAT replay

Example:
  FRONTEND_HOST=http://localhost:80 \\
  LOAD_USERS_SERIES="100 200 300 400" \\
  REPEATS=3 DURATION=300 \\
  MODEL_PATH=~/adaptive_tracer/logs/lstm_multitask_cats5_seq512_382061/model_best.pt \\
  VOCAB_PATH=~/adaptive_tracer/micro-service-trace-data/preprocessed_seq512/preprocessed_lmat_kernel_cats5_seq512/vocab.pkl \\
  DELAY_PATH=~/adaptive_tracer/micro-service-trace-data/preprocessed_seq512/preprocessed_lmat_kernel_cats5_seq512/delay_spans.pkl \\
  $0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "$EXPERIMENT_ROOT"

MANIFEST="$EXPERIMENT_ROOT/run_manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
    echo "condition,users,repeat,run_id,duration_s,think_min,think_max,frontend_host,status,started_utc,ended_utc" > "$MANIFEST"
fi

echo "============================================================"
echo "Reviewer Overhead Matrix"
echo "Project        : $PROJECT_DIR"
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Host           : $FRONTEND_HOST"
echo "Users          : $LOAD_USERS_SERIES"
echo "Repeats        : $REPEATS"
echo "Duration       : ${DURATION}s"
echo "Think time     : ${THINK_MIN}-${THINK_MAX}s"
echo "Conditions     : $CONDITIONS"
echo "Torch threads  : $TORCH_THREADS"
echo "============================================================"

run_condition() {
    local condition="$1"
    local run_id="$2"
    local users="$3"
    local repeat="$4"
    local started ended status

    started=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    status="ok"

    echo
    echo ">>> condition=$condition users=$users repeat=$repeat run_id=$run_id"

    export EXPERIMENT_ROOT FRONTEND_HOST THINK_MIN THINK_MAX LOAD_USERS TORCH_THREADS
    export MODEL_PATH VOCAB_PATH DELAY_PATH MODEL_TYPE
    export N_HIDDEN N_LAYER N_HEAD DIM_SYS DIM_ENTRY DIM_RET DIM_PROC DIM_PID DIM_TID DIM_ORDER DIM_TIME N_CATEGORIES
    LOAD_USERS="$users"

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
        lmat_sync)
            "$SCRIPT_DIR/lmat_sync_run.sh" "$run_id" "$DURATION" "$QUIET_FLAG"
            ;;
        *)
            echo "Unknown condition: $condition" >&2
            return 1
            ;;
    esac

    ended=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "$condition,$users,$repeat,$run_id,$DURATION,$THINK_MIN,$THINK_MAX,$FRONTEND_HOST,$status,$started,$ended" >> "$MANIFEST"
}

for users in $LOAD_USERS_SERIES; do
    for repeat in $(seq 1 "$REPEATS"); do
        run_id=$(printf "u%03d_r%02d" "$users" "$repeat")
        for condition in $CONDITIONS; do
            run_condition "$condition" "$run_id" "$users" "$repeat"
            if [[ "$COOLDOWN_S" -gt 0 ]]; then
                echo "Cooling down for ${COOLDOWN_S}s ..."
                sleep "$COOLDOWN_S"
            fi
        done
    done
done

echo
echo "All reviewer-overhead runs completed."
echo "Manifest: $MANIFEST"
echo
echo "Next step:"
echo "python3 \"$SCRIPT_DIR/analyse_reviewer_overhead.py\" --experiment_root \"$EXPERIMENT_ROOT\" --reference_users 200"
