#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=adaptive_tracer_1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

# Go to your project directory in SCRATCH (good: writable on compute nodes)
cd /scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer

# Make sure logs folder exists (in scratch)
mkdir -p logs

module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

# Activate your scratch-based venv from this project
source /scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/.venv/bin/activate

# Optional: confirm GPU is visible
srun nvidia-smi

# Run (use 0 for single GPU)
srun python main.py --log_folder logs/lstm-1 --data_path /scratch/yuvraj17/adaptive_tracing_scratch/trace_data --train_folder "Train:train_id" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket" --generate_dataset --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --model lstm --n_hidden 256 --n_layer 2 --dim_sys 48 --dim_proc 48 --dim_entry 12 --dim_ret 12 --dim_pid 12 --dim_tid 12 --dim_time 12 --dim_order 12 --dim_f_mean 0 --optimizer adam --n_update 1000000 --eval 1000 --lr 0.001 --ls 0.1 --batch 16 --gpu "0" --amp --reduce_lr_patience 5 --early_stopping_patience 20 --dropout 0.01 --clip 10 --analysis --seed 1 --n_categories 6 --train_event_model --train_latency_model