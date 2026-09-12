#!/bin/bash
#SBATCH --job-name=txt_dump_to_shards_7cat_4096_5000
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TXT_DUMP_DIR=$SCRATCH/micro-service-trace-data-txt-dump
TRACE_ROOT=$TXT_DUMP_DIR
OUTPUT_DIR=$SCRATCH/micro-service-trace-data/preprocessed_lmat_kernel_cats7_seq4096_5000
LOG_DIR=$PROJECT/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PROJECT"

source "$PROJECT/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Job            : txt_dump_to_shards_cats7_seq4096_5000"
echo "Job ID         : ${SLURM_JOB_ID:-manual}"
echo "Node           : ${SLURMD_NODENAME:-unknown}"
echo "Project        : $PROJECT"
echo "Trace root     : $TRACE_ROOT"
echo "Text dumps     : $TXT_DUMP_DIR"
echo "Output dir     : $OUTPUT_DIR"
echo "Paper bins     : 7"
echo "Code cats      : 8"
echo "Max seq len    : 4096"
echo "Shard size     : 5000"
echo "CPUs           : ${SLURM_CPUS_PER_TASK:-8}"
echo "============================================================"

if [[ ! -d "$TXT_DUMP_DIR" ]]; then
  echo "ERROR: txt dump dir not found: $TXT_DUMP_DIR" >&2
  exit 1
fi

if [[ ! -f "$PROJECT/microservice/preprocess_lmat_kernel.py" ]]; then
  echo "ERROR: missing preprocessor: $PROJECT/microservice/preprocess_lmat_kernel.py" >&2
  exit 1
fi

echo "[1/3] Python sanity"
python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as exc:
    print("ERROR importing numpy:", exc)
    raise
PY

echo "[2/3] Txt dump layout"
ls -lah "$TXT_DUMP_DIR"

echo "[3/3] Building LMAT shards from txt dumps"
python -u microservice/preprocess_lmat_kernel.py \
  --trace_root "$TRACE_ROOT" \
  --txt_dump_dir "$TXT_DUMP_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --window_ms 100 \
  --warmup_s 5 \
  --min_events 8 \
  --max_seq_len 4096 \
  --paper_duration_bins 7 \
  --shard_size 5000 \
  --event_scope syscall \
  --normal_dir normal \
  --cpu_dir anomaly_cpu \
  --disk_dir anomaly_disk \
  --mem_dir anomaly_mem \
  --net_dir anomaly_net \
  --normal_train_runs run01,run02,run03 \
  --normal_valid_runs run04 \
  --normal_test_runs run05 \
  --ood_valid_runs run04 \
  --ood_test_runs run05

EXIT_CODE=$?
echo "============================================================"
echo "Finished with exit code: $EXIT_CODE"
echo "Output dir: $OUTPUT_DIR"
echo "Manifest  : $OUTPUT_DIR/dataset_manifest.json"
echo "Vocab     : $OUTPUT_DIR/vocab.pkl"
echo "Delays    : $OUTPUT_DIR/delay_spans.pkl"
echo "============================================================"
exit $EXIT_CODE
