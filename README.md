# PySR on AI-Feynman Data

This repository runs [PySR](https://ai.damtp.cam.ac.uk/pysr/dev/) against AI-Feynman-style datasets.

PySR is a high-performance symbolic regression library built on top of the Julia backend `SymbolicRegression.jl`. The AI-Feynman repository documents its data files as plain numeric tables with one column per variable, separated by spaces, commas, or tabs.

## What this repo does

- Loads AI-Feynman-style data files with no header row
- Lets you choose which column is the target
- Runs `PySRRegressor` with a practical default operator set
- Saves the discovered Pareto-front equations and run metadata

## Files

- `symbolic_regression.py`: CLI entry point
- `run_benchmarks.py`: batch benchmark runner across datasets and seeds
- `plot_benchmarks.py`: quick plotting utility for aggregate benchmark CSVs
- `requirements.txt`: Python dependencies

## Install with Conda

Create the project environment:

```bash
conda env create -f environment.yml
conda activate pysr-aifeynman
```

Install note:

- `environment.yml` installs the Python stack with Conda and `pysr` via `pip`.
- On first import, PySR sets up its Julia dependencies automatically.
- If Julia is already installed and you want PySR to use it, set `JULIA_EXE` before running.

## Run on AI-Feynman data

If you already have a local AI-Feynman dataset file, run:

```bash
python3 symbolic_regression.py \
  --data /path/to/example1.txt \
  --vars-name x0 x1 x2 x3 y \
  --niterations 100 \
  --population-size 64 \
  --populations 12
```

The default assumption is that the last column is the target. If your target is somewhere else, pass `--target-column`.

## Curriculum over dataset size

For quicker MVP experiments on large Feynman files, you can run PySR on progressively larger nested subsets:

```bash
python3 symbolic_regression.py \
  --data /path/to/example1.txt \
  --vars-name x0 x1 y \
  --curriculum-rows 1000 10000 100000 full \
  --niterations 20 \
  --population-size 32 \
  --populations 4
```

This reuses the same PySR model with warm start across stages and writes one output folder per stage, plus a `curriculum_summary.json` file under the run directory.

Example with a non-final target column:

```bash
python3 symbolic_regression.py \
  --data /path/to/data.txt \
  --target-column 0 \
  --vars-name y x0 x1 x2
```

## Dry run

Use this to verify parsing and column naming before starting a PySR search:

```bash
python3 symbolic_regression.py \
  --data /path/to/example1.txt \
  --vars-name x0 x1 x2 x3 y \
  --dry-run
```

## Outputs

Each run writes to `outputs/<run-id>/`:

- `equations.csv`: the equations table exported from PySR
- `run_metadata.json`: dataset info, chosen settings, and the best expression
- `stage_results.csv`: one flat row per training stage for plotting and aggregation

Batch runs created by `run_benchmarks.py` also write:

- `outputs/benchmark_results.csv`: aggregate CSV across all datasets and seeds
- per-run stage folders and metadata under `outputs/<dataset>_seed<seed>/`

## Benchmark multiple datasets

Run a small sweep over several Feynman files and seeds:

```bash
python3 run_benchmarks.py \
  --datasets Feynman_without_units/I.6.2 Feynman_without_units/I.12.1 \
  --seeds 0 1 2 \
  --curriculum-rows 1000 10000 100000 full \
  --niterations 20 \
  --population-size 32 \
  --populations 4
```

You can also point it at a text file with one dataset path per line:

```bash
python3 run_benchmarks.py \
  --dataset-list cluster/datasets_example.txt \
  --seeds 0 1 2
```

## Create plots

After a benchmark sweep, generate a few quick plots:

```bash
python3 plot_benchmarks.py --results-csv outputs/benchmark_results.csv
```

This writes PNG files under `plots/`:

- `runtime_vs_full_mse.png`
- `curriculum_stage_vs_full_mse.png`
- `final_stage_boxplot.png`

## Cluster usage

The `cluster/` directory includes Slurm templates for Purdue Scholar-style workflows:

- `cluster/run_single_benchmark.slurm`: run one dataset benchmark job
- `cluster/run_array_benchmarks.slurm`: run one dataset per Slurm array task
- `cluster/datasets_example.txt`: sample dataset list for array jobs

Example single-job submission:

```bash
sbatch --export=PROJECT_DIR=$HOME/pysr-feynman,DATASET_PATH=Feynman_without_units/I.6.2 cluster/run_single_benchmark.slurm
```

Example array submission:

```bash
sbatch --array=0-2 --export=PROJECT_DIR=$HOME/pysr-feynman,DATASET_LIST=cluster/datasets_example.txt cluster/run_array_benchmarks.slurm
```

For a fast MVP workflow similar to your course skeleton, this repo also includes:

- `scholar.sh`: a CPU-oriented Slurm wrapper that activates the Conda env
- `interface.sh`: submits baseline and curriculum jobs for each dataset you name

Example:

```bash
chmod +x interface.sh scholar.sh
./interface.sh I.6.2 I.12.1 I.16.6
```

That submits two jobs per dataset:

- baseline PySR run
- curriculum PySR run

Useful environment overrides:

```bash
CPUS=8 NITERATIONS=20 POPULATION_SIZE=32 POPULATIONS=4 ./interface.sh I.6.2 I.12.1
```

By default, `interface.sh` assumes:

- project checkout at `$HOME/pysr-feynman`
- extracted data under `data/Feynman_without_units/`
- one seed (`0`)
- baseline outputs in `outputs/baseline/`
- curriculum outputs in `outputs/curriculum/`

## Notes on the sources

- PySR docs: `pip install pysr`, and Julia dependencies are installed at first import.
- AI-Feynman docs: dataset files are plain numeric tables with columns separated by spaces, commas, or tabs.
