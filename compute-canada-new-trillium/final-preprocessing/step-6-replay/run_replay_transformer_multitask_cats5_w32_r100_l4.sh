#!/bin/bash
#SBATCH --job-name=atr_trf_c5_w32r100l4
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
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
MODEL=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/transformer_multitask_cats5_seq512_382062/model_best.pt
OUTPUT_DIR=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/adaptive_trace_replay_transformer_multitask_cats5_382062_w32_r100_l4

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
echo "Mode         : Adaptive Replay | Transformer | cats5 | w32 r1.00 l4"
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

echo "[4/4] Adaptive tracing replay"
python -u microservice/adaptive_trace_replay.py \
  --preprocessed_dir "$DATA" \
  --load_model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --model transformer \
  --n_categories 6 \
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
  --train_event_model \
  --train_latency_model \
  --multitask_lambda 0.5 \
  --batch 64 \
  --num_workers 2 \
  --progress_every_batches 100 \
  --amp \
  --window_size 32 \
  --trigger_ratio 1.0 \
  --warmup_sequences 256 \
  --linger_sequences 4 \
  --scenario_mode both

EXIT_CODE=$?
echo "Adaptive replay finished with exit code: $EXIT_CODE"
echo "Results: $OUTPUT_DIR/adaptive_trace_replay_results.json"
echo "Summary: $OUTPUT_DIR/adaptive_trace_replay_summary.csv"
exit $EXIT_CODE
