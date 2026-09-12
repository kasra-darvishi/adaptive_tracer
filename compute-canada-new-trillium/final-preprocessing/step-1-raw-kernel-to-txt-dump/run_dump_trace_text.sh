#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=dump_trace_text
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data
DUMP_ROOT=$SCRATCH/micro-service-trace-data/micro-service-trace-data-txt-dump

SCENARIOS=(normal anomaly_cpu anomaly_disk anomaly_mem anomaly_net)
RUNS=(run01 run02 run03 run04 run05)

mkdir -p "$DUMP_ROOT" "$PROJECT/logs"
cd "$PROJECT"

module purge
module load StdEnv/2023
module load python/3.11.5
module load babeltrace2

echo "============================================================"
echo "Job          : dump_trace_text"
echo "Trace root   : $TRACE_ROOT"
echo "Dump root    : $DUMP_ROOT"
echo "Scenarios    : ${SCENARIOS[*]}"
echo "Runs         : ${RUNS[*]}"
echo "============================================================"

if [[ ! -d "$TRACE_ROOT" ]]; then
  echo "ERROR: trace root not found: $TRACE_ROOT" >&2
  exit 1
fi

echo "[1/2] Trace root listing"
srun bash -lc "ls -lah '$TRACE_ROOT' | head -50"

echo "[2/2] Generating text dumps"
for scenario in "${SCENARIOS[@]}"; do
  for run in "${RUNS[@]}"; do
    RUN_DIR="$TRACE_ROOT/$scenario/$run"
    OUT_DIR="$DUMP_ROOT/$scenario/$run"
    mkdir -p "$OUT_DIR"

    if [[ -d "$RUN_DIR/kernel" ]]; then
      echo "[KERNEL] $scenario/$run -> $OUT_DIR/kernel.txt"
      srun --ntasks=1 bash -lc "babeltrace2 '$RUN_DIR/kernel' > '$OUT_DIR/kernel.txt'"
    else
      echo "[WARN] Missing kernel trace: $RUN_DIR/kernel"
    fi

    if [[ -d "$RUN_DIR/ust" ]]; then
      echo "[UST]    $scenario/$run -> $OUT_DIR/ust.txt"
      srun --ntasks=1 bash -lc "babeltrace2 '$RUN_DIR/ust' > '$OUT_DIR/ust.txt'"
    else
      echo "[WARN] Missing ust trace: $RUN_DIR/ust"
    fi
  done
done

echo "============================================================"
echo "Done. Dumps written to: $DUMP_ROOT"
echo "============================================================"
