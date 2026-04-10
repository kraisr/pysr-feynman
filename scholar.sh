#!/bin/bash

#SBATCH --export=ALL
#SBATCH -A scholar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH -J pysr-feynman
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail

module load conda/2024.09
cd "${PROJECT_DIR:-$HOME/pysr-feynman}"
conda activate ./env

mkdir -p slurm_logs

srun "$@"
