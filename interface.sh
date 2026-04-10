#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/testing/pysr-feynman}"
DATA_ROOT="${DATA_ROOT:-data/Feynman_without_units}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SEED="${SEED:-0}"
CPUS="${CPUS:-4}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
NITERATIONS="${NITERATIONS:-10}"
POPULATION_SIZE="${POPULATION_SIZE:-24}"
POPULATIONS="${POPULATIONS:-3}"
BASELINE_OUT="${BASELINE_OUT:-${OUTPUT_ROOT}/baseline}"
CURRICULUM_OUT="${CURRICULUM_OUT:-${OUTPUT_ROOT}/curriculum}"
CURRICULUM_ROWS="${CURRICULUM_ROWS:-1000 10000 full}"

if [[ $# -eq 0 ]]; then
    echo "Usage: ./interface.sh dataset1 [dataset2 ...]"
    echo "Example: ./interface.sh I.6.2 I.12.1 I.16.6"
    exit 1
fi

mkdir -p slurm_logs

for dataset_name in "$@"; do
    dataset_path="${DATA_ROOT}/${dataset_name}"

    sbatch \
        --time="${TIME_LIMIT}" \
        --cpus-per-task="${CPUS}" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
        scholar.sh \
        python run_benchmarks.py \
        --datasets "${dataset_path}" \
        --seeds "${SEED}" \
        --niterations "${NITERATIONS}" \
        --population-size "${POPULATION_SIZE}" \
        --populations "${POPULATIONS}" \
        --procs "${CPUS}" \
        --output-dir "${BASELINE_OUT}"

    sbatch \
        --time="${TIME_LIMIT}" \
        --cpus-per-task="${CPUS}" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
        scholar.sh \
        python run_benchmarks.py \
        --datasets "${dataset_path}" \
        --seeds "${SEED}" \
        --curriculum-rows ${CURRICULUM_ROWS} \
        --niterations "${NITERATIONS}" \
        --population-size "${POPULATION_SIZE}" \
        --populations "${POPULATIONS}" \
        --procs "${CPUS}" \
        --output-dir "${CURRICULUM_OUT}"
done
