#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=build_lmat_shards_c6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data
OUTPUT_DIR=$SCRATCH/micro-service-trace-data/preprocessed_lmat_kernel_cats6
TXT_DUMP=$SCRATCH/micro-service-trace-data/micro-service-trace-data-txt-dump

WINDOW_MS=100
WARMUP_S=5
MIN_EVENTS=8
MAX_SEQ_LEN=512
N_CATEGORIES=6
SHARD_SIZE=5000
EVENT_SCOPE=syscall

NORMAL_TRAIN=run01,run02,run03
NORMAL_VALID=run04
NORMAL_TEST=run05
OOD_VALID=run04
OOD_TEST=run05

mkdir -p "$OUTPUT_DIR" "$PROJECT/logs"
cd "$PROJECT"

module purge
module load StdEnv/2023
module load python/3.11.5
module load babeltrace2 2>/dev/null || true

source "$PROJECT/.venv/bin/activate"

echo "============================================================"
echo "Job            : build_lmat_shards_c6"
echo "Project        : $PROJECT"
echo "Trace root     : $TRACE_ROOT"
echo "Output dir     : $OUTPUT_DIR"
echo "Text dumps     : $TXT_DUMP"
echo "Event scope    : $EVENT_SCOPE"
echo "Window (ms)    : $WINDOW_MS"
echo "Warmup (s)     : $WARMUP_S"
echo "Min events     : $MIN_EVENTS"
echo "Max seq len    : $MAX_SEQ_LEN"
echo "Categories     : $N_CATEGORIES"
echo "Shard size     : $SHARD_SIZE"
echo "============================================================"

if [[ ! -d "$TRACE_ROOT" ]]; then
  echo "ERROR: trace root not found: $TRACE_ROOT" >&2
  exit 1
fi

if [[ ! -f "$PROJECT/microservice/preprocess_lmat_kernel.py" ]]; then
  echo "ERROR: preprocessor not found under $PROJECT/microservice" >&2
  exit 1
fi

echo "[1/3] Python sanity"
srun python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as exc:
    print("ERROR importing numpy:", exc)
    raise
PY

echo "[2/3] Trace root listing"
srun bash -lc "ls -lah '$TRACE_ROOT' | head -50"

echo "[3/3] Building LMAT shards"
srun python -u microservice/preprocess_lmat_kernel.py \
  --trace_root "$TRACE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --txt_dump_dir "$TXT_DUMP" \
  --window_ms "$WINDOW_MS" \
  --warmup_s "$WARMUP_S" \
  --min_events "$MIN_EVENTS" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --n_categories "$N_CATEGORIES" \
  --shard_size "$SHARD_SIZE" \
  --event_scope "$EVENT_SCOPE" \
  --normal_dir normal \
  --cpu_dir anomaly_cpu \
  --disk_dir anomaly_disk \
  --mem_dir anomaly_mem \
  --net_dir anomaly_net \
  --normal_train_runs "$NORMAL_TRAIN" \
  --normal_valid_runs "$NORMAL_VALID" \
  --normal_test_runs "$NORMAL_TEST" \
  --ood_valid_runs "$OOD_VALID" \
  --ood_test_runs "$OOD_TEST"

EXIT_CODE=$?
echo "============================================================"
echo "Finished with exit code: $EXIT_CODE"
echo "Output dir: $OUTPUT_DIR"
echo "Manifest  : $OUTPUT_DIR/dataset_manifest.json"
echo "Vocab     : $OUTPUT_DIR/vocab.pkl"
echo "Delays    : $OUTPUT_DIR/delay_spans.pkl"
echo "============================================================"
exit $EXIT_CODE
