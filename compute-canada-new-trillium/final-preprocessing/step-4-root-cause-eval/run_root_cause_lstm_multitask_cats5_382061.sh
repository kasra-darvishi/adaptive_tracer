#!/bin/bash
#SBATCH --job-name=rca_lstm_c5_382061
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

BASE_SCRATCH=${SCRATCH:-$HOME/scratch}
WORKROOT=$BASE_SCRATCH/adaptive_tracing_scratch
PROJECT=$WORKROOT/adaptive_tracer
DATA=/scratch/yuvraj17/adaptive_tracing_scratch/micro-service-trace-data/preprocessed_lmat_kernel_cats5_seq512
MODEL=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/lstm_multitask_cats5_seq512_382061/model_best.pt
OUTPUT_DIR=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/root_cause_lstm_multitask_cats5_382061_hdbscan

mkdir -p "$OUTPUT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline
export TRITON_CACHE_DIR="$WORKROOT/.triton_cache"
export TORCH_HOME="$WORKROOT/.torch"
export HF_HOME="$WORKROOT/.hf_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME"

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Mode         : Root Cause Eval | LSTM | Multitask | cats5"
echo "Project      : $PROJECT"
echo "Data dir     : $DATA"
echo "Model        : $MODEL"
echo "Output dir   : $OUTPUT_DIR"
echo "============================================================"

source "$PROJECT/.venv/bin/activate"
cd "$PROJECT"

echo "[1/4] Module list"
module list

echo "[2/4] GPU sanity"
srun nvidia-smi

echo "[3/4] Python + torch sanity"
srun python - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "[4/4] Root-cause evaluation"
python -u microservice/run_root_cause_eval.py \
  --preprocessed_dir "$DATA" \
  --load_model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --model lstm \
  --n_categories 6 \
  --max_seq_len 512 \
  --n_hidden 1024 \
  --n_layer 6 \
  --dim_sys 48 \
  --dim_entry 12 \
  --dim_ret 12 \
  --dim_proc 48 \
  --dim_pid 12 \
  --dim_tid 12 \
  --dim_order 12 \
  --dim_time 12 \
  --train_event_model \
  --train_latency_model \
  --multitask_lambda 0.5 \
  --batch 64 \
  --num_workers 2 \
  --amp \
  --combine_strategy mean \
  --centroid_source all \
  --cluster_method hdbscan \
  --cluster_metric euclidean \
  --cluster_min_size 128 \
  --cluster_max_records_per_label 25000

EXIT_CODE=$?
echo "Root-cause evaluation finished with exit code: $EXIT_CODE"
echo "Results: $OUTPUT_DIR/root_cause_results.json"
echo "Predictions: $OUTPUT_DIR/root_cause_predictions.csv"
exit $EXIT_CODE
