#!/bin/bash
#SBATCH -A scholar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=pysr-array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load conda/2024.09
cd "${PROJECT_DIR:-$HOME/pysr-feynman}"
conda activate ./env

mkdir -p logs

DATASET_LIST="${DATASET_LIST:-cluster/datasets_example.txt}"
DATASET_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$DATASET_LIST")"

python run_benchmarks.py \
  --datasets "$DATASET_PATH" \
  --seeds ${SEEDS:-0 1 2} \
  --curriculum-rows ${CURRICULUM_ROWS:-1000 10000 100000 full} \
  --niterations ${NITERATIONS:-20} \
  --population-size ${POPULATION_SIZE:-32} \
  --populations ${POPULATIONS:-4} \
  --procs ${SLURM_CPUS_PER_TASK:-8} \
  --output-dir "${OUTPUT_DIR:-outputs}"
