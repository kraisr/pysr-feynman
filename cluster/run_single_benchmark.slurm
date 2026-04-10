#!/bin/bash
#SBATCH -A scholar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=pysr-feynman
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load conda/2024.09
cd "${PROJECT_DIR:-$HOME/pysr-feynman}"
conda activate ./env

mkdir -p logs

python run_benchmarks.py \
  --datasets "${DATASET_PATH:?Set DATASET_PATH before sbatch}" \
  --seeds ${SEEDS:-0 1 2} \
  --curriculum-rows ${CURRICULUM_ROWS:-1000 10000 100000 full} \
  --niterations ${NITERATIONS:-20} \
  --population-size ${POPULATION_SIZE:-32} \
  --populations ${POPULATIONS:-4} \
  --procs ${SLURM_CPUS_PER_TASK:-8} \
  --output-dir "${OUTPUT_DIR:-outputs}"
