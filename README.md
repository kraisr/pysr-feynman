# PySR on AI-Feynman Data

This repository runs [PySR](https://ai.damtp.cam.ac.uk/pysr/dev/) on AI-Feynman-style datasets and compares a full-data baseline against a curriculum schedule.

<img width="197" height="600" alt="workflow" src="https://github.com/user-attachments/assets/7bb4095a-70d4-487a-8c75-cca4c3e05357" />

## Overview

- `symbolic_regression.py`: run PySR on one dataset
- `run_benchmarks.py`: run multiple datasets
- `merge_stage_results.py`: merge per-run `stage_results.csv` files
- `plot_comparison.py`: create side-by-side runtime and MSE plots
- `interface.sh` and `scholar.sh`: simple Scholar/Slurm submission workflow

## Setup

```bash
conda env create -f environment.yml
conda activate pysr-feynman
```

On first import, PySR installs its Julia-side dependencies automatically.

## Run One Dataset

Baseline:

```bash
python3 symbolic_regression.py \
  --data /path/to/data.txt \
  --vars-name x0 x1 y \
  --niterations 20 \
  --population-size 32 \
  --populations 4
```

Curriculum:

```bash
python3 symbolic_regression.py \
  --data /path/to/data.txt \
  --vars-name x0 x1 y \
  --curriculum-rows 1000 10000 full \
  --niterations 20 \
  --population-size 32 \
  --populations 4
```

Dry run:

```bash
python3 symbolic_regression.py \
  --data /path/to/data.txt \
  --vars-name x0 x1 y \
  --dry-run
```

If the target is not the last column, pass `--target-column`.

## Benchmark Workflow

Run a small sweep:

```bash
python3 run_benchmarks.py \
  --datasets data/Feynman_without_units/I.6.2 data/Feynman_without_units/I.12.1 \
  --seeds 0 \
  --curriculum-rows 1000 10000 full \
  --niterations 10 \
  --population-size 24 \
  --populations 3
```

Merge results:

```bash
python3 merge_stage_results.py --input-dir outputs/baseline
python3 merge_stage_results.py --input-dir outputs/curriculum
```

Create comparison plots:

```bash
python3 plot_comparison.py \
  --baseline-csv outputs/baseline/benchmark_results_merged.csv \
  --curriculum-csv outputs/curriculum/benchmark_results_merged.csv \
  --output-dir plots/comparison
```

This writes:

- `final_mse_comparison.png`
- `runtime_comparison.png`
- `comparison_summary.csv`

## Outputs

Each run writes:

- `outputs/<run-id>/stage_results.csv`

The merged workflow writes:

- `outputs/<method>/benchmark_results_merged.csv`
- comparison plots under your chosen output directory

## Scholar / Slurm

Submit baseline and curriculum jobs for selected datasets:

```bash
chmod +x interface.sh scholar.sh
PROJECT_DIR=/home/<username>/pysr-feynman ./interface.sh I.6.2 I.12.1 I.16.6
```

Useful overrides:

```bash
CPUS=8 NITERATIONS=10 POPULATION_SIZE=24 POPULATIONS=3 ./interface.sh I.6.2 I.12.1
```

By default, `interface.sh` expects:

- the repo at `$HOME/pysr-feynman`
- extracted data under `data/Feynman_without_units/`
- baseline outputs under `outputs/baseline/`
- curriculum outputs under `outputs/curriculum/`

## Data Format

AI-Feynman files are plain numeric tables with no header row. Columns can be separated by spaces, commas, or tabs.
