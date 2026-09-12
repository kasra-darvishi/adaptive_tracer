#!/bin/bash
#SBATCH --account=def-naser2

# RESSOURCES
#SBATCH --cpus-per-task=10		# Number of CPUs
#SBATCH --mem=32000M			# Memory
#SBATCH --gres=gpu:v100:4		# Number of GPUs
#SBATCH --time=04-00:00		    # Brackets: 3h, 12h, 1d, 3d, 7d

# JOB SPECIFICATION
#SBATCH --job-name=lstm-2
#SBATCH --output=/home/kasra/adaptive-trace/logs/%x-%j

# LOAD VIRTUAL ENVIRONMENT
source ~/venv/bin/activate

# TASK
cd /home/kasra/adaptive-trace
python main.py --log_folder logs/lstm-2 --data_path /scratch/yuvraj17/adaptive_tracing_scratch/trace_data --train_folder "Train:train_id" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket,Valid OOD (SSL):valid_ood_ssl" --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --model lstm --n_hidden 256 --n_layer 2 --dim_sys 48 --dim_proc 48 --dim_entry 12 --dim_ret 12 --dim_pid 12 --dim_tid 12 --dim_time 12 --dim_order 12 --dim_f_mean 0 --optimizer adam --n_update 1000000 --eval 1000 --lr 0.001 --ls 0.1 --batch 16 --gpu "0,1,2,3" --amp --reduce_lr_patience 5 --early_stopping_patience 20 --dropout 0.01 --clip 10 --analysis --seed 2 --n_categories 6 --train_event_model --train_latency_model
