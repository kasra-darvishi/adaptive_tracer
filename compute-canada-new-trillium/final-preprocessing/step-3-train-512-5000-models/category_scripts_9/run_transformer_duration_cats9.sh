#!/bin/bash
#SBATCH --job-name=trf_duration_cats9_512
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=09:00:00
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
DATA=$WORKROOT/micro-service-trace-data/preprocessed_lmat_kernel_cats9_seq512
LOG_DIR=$PROJECT/logs/transformer_duration_cats9_seq512_${SLURM_JOB_ID}

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline
export WANDB_DIR="$LOG_DIR"
export WANDB_CACHE_DIR="$WORKROOT/wandb_cache"
export TRITON_CACHE_DIR="$WORKROOT/.triton_cache"
export TORCH_HOME="$WORKROOT/.torch"
export HF_HOME="$WORKROOT/.hf_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME" "$WANDB_CACHE_DIR"

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Mode         : Transformer | Duration-only | paper bins=9 | seq=512"
echo "Project      : $PROJECT"
echo "Data dir     : $DATA"
echo "Log dir      : $LOG_DIR"
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

python -u microservice/train_sockshop.py \
  --preprocessed_dir "$DATA" \
  --model transformer \
  --n_categories 10 \
  --max_seq_len 512 \
  --n_head 8 \
  --n_hidden 1024 \
  --n_layer 6 \
  --dropout 0.1 \
  --activation gelu \
  --dim_sys 48 \
  --dim_entry 12 \
  --dim_ret 12 \
  --dim_proc 48 \
  --dim_pid 12 \
  --dim_tid 12 \
  --dim_order 12 \
  --dim_time 12 \
  --train_latency_model \
  --batch 64 \
  --accum_steps 1 \
  --n_epochs 30 \
  --early_stopping_patience 10 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --clip 1.0 \
  --num_workers 8 \
  --label_smoothing 0.0 \
  --amp \
  --eval_every 2000 \
  --save_every 10000 \
  --wandb_project sockshop_lmat \
  --wandb_run_name "transformer_duration_cats9_seq512_${SLURM_JOB_ID}" \
  --log_dir "$LOG_DIR" \
  --gpu 0

EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE"
echo "OOD results: $LOG_DIR/ood_results.json"
wandb sync "$LOG_DIR/wandb/latest-run" 2>/dev/null || true
exit $EXIT_CODE
