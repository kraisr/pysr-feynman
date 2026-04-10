#!/usr/bin/env python3
"""Run a small batch of PySR benchmarks and aggregate CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from symbolic_regression import run_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PySR runs across datasets and seeds.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Explicit dataset paths to run.",
    )
    parser.add_argument(
        "--dataset-list",
        help="Optional text file with one dataset path per line.",
    )
    parser.add_argument(
        "--dataset-glob",
        nargs="*",
        default=[],
        help="Optional glob patterns to expand into datasets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds to benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Where run artifacts and aggregate CSV files are written.",
    )
    parser.add_argument(
        "--aggregate-name",
        default="benchmark_results.csv",
        help="Filename for the aggregated benchmark CSV under output-dir.",
    )

    # Reuse the training knobs from symbolic_regression.py by parsing
    # only the known arguments we care about after this script consumes
    # benchmark-specific flags.
    parser.add_argument("--target-column", type=int, default=-1)
    parser.add_argument("--niterations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--maxsize", type=int, default=30)
    parser.add_argument("--maxdepth", type=int, default=12)
    parser.add_argument("--procs", type=int, default=0)
    parser.add_argument("--precision", type=int, default=64, choices=[32, 64])
    parser.add_argument(
        "--model-selection",
        default="best",
        choices=["best", "accuracy", "score"],
    )
    parser.add_argument(
        "--binary-operators",
        nargs="+",
        default=["+", "-", "*", "/"],
    )
    parser.add_argument(
        "--unary-operators",
        nargs="+",
        default=["sin", "cos", "exp", "log", "sqrt", "square"],
    )
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--select-k-features", type=int)
    parser.add_argument("--curriculum-rows", nargs="+")
    return parser.parse_args()


def load_dataset_paths(args: argparse.Namespace) -> list[Path]:
    dataset_paths: list[Path] = []
    for dataset in args.datasets:
        dataset_paths.append(Path(dataset).expanduser())
    if args.dataset_list:
        list_path = Path(args.dataset_list).expanduser()
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                dataset_paths.append(Path(stripped).expanduser())
    for pattern in args.dataset_glob:
        dataset_paths.extend(sorted(Path().glob(pattern)))

    resolved = []
    seen: set[Path] = set()
    for path in dataset_paths:
        resolved_path = path.resolve()
        if resolved_path not in seen:
            resolved.append(resolved_path)
            seen.add(resolved_path)
    if not resolved:
        raise SystemExit("No datasets selected. Use --datasets, --dataset-list, or --dataset-glob.")
    return resolved


def build_run_args(base: argparse.Namespace, dataset_path: Path, seed: int) -> argparse.Namespace:
    run_id = f"{dataset_path.stem}_seed{seed}"
    base_output_dir = Path(base.output_dir).expanduser().resolve()
    return argparse.Namespace(
        data=str(dataset_path),
        target_column=base.target_column,
        vars_name=None,
        output_dir=str(base_output_dir),
        run_id=run_id,
        niterations=base.niterations,
        population_size=base.population_size,
        populations=base.populations,
        maxsize=base.maxsize,
        maxdepth=base.maxdepth,
        procs=base.procs,
        precision=base.precision,
        model_selection=base.model_selection,
        binary_operators=base.binary_operators,
        unary_operators=base.unary_operators,
        random_state=seed,
        denoise=base.denoise,
        select_k_features=base.select_k_features,
        dry_run=False,
        curriculum_rows=base.curriculum_rows,
    )


def write_aggregate_csv(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_paths = load_dataset_paths(args)
    all_rows: list[dict[str, object]] = []

    for dataset_path in dataset_paths:
        for seed in args.seeds:
            print(f"\n### benchmark dataset={dataset_path.name} seed={seed}")
            run_args = build_run_args(args, dataset_path, seed)
            stage_results = run_from_args(run_args)
            for result in stage_results:
                all_rows.append(result.__dict__)

    output_dir = Path(args.output_dir).expanduser().resolve()
    aggregate_path = output_dir / args.aggregate_name
    write_aggregate_csv(all_rows, aggregate_path)
    print(f"\naggregate_csv={aggregate_path}")


if __name__ == "__main__":
    main()
