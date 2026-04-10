#!/usr/bin/env python3
"""Run PySR on AI-Feynman-style symbolic regression datasets."""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Missing dependency: numpy. Install the project dependencies with "
        "`python3 -m pip install -r requirements.txt` and rerun."
    ) from exc


DEFAULT_BINARY_OPERATORS = ["+", "-", "*", "/"]
DEFAULT_UNARY_OPERATORS = ["sin", "cos", "exp", "log", "sqrt", "square"]
SPLIT_PATTERN = re.compile(r"[\s,]+")


@dataclass
class DatasetBundle:
    path: Path
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    n_rows: int
    n_columns: int


@dataclass
class CurriculumStage:
    index: int
    label: str
    row_count: int
    X: np.ndarray
    y: np.ndarray


@dataclass
class RunStageResult:
    run_id: str
    dataset_name: str
    dataset_path: str
    seed: int
    stage_index: int
    stage_name: str
    stage_label: str
    stage_rows: int
    total_rows: int
    best_equation: str
    latex: str
    runtime_sec: float
    stage_mse: float
    stage_mae: float
    full_mse: float
    full_mae: float
    complexity: float | None
    loss: float | None
    score: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PySR on AI-Feynman-format data files.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to an AI-Feynman-style data file with numeric columns and no header.",
    )
    parser.add_argument(
        "--target-column",
        type=int,
        default=-1,
        help="Zero-based target column index. Defaults to the last column.",
    )
    parser.add_argument(
        "--vars-name",
        nargs="+",
        help="Optional variable names for all columns, including the target column.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where PySR outputs and metadata will be written.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional PySR run identifier. Defaults to the dataset stem.",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=100,
        help="Number of PySR iterations.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=64,
        help="Population size per population.",
    )
    parser.add_argument(
        "--populations",
        type=int,
        default=12,
        help="Number of populations.",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=30,
        help="Maximum PySR equation complexity.",
    )
    parser.add_argument(
        "--maxdepth",
        type=int,
        default=12,
        help="Maximum PySR tree depth.",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=0,
        help="Number of worker processes. Use 0 to let PySR choose.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=64,
        choices=[32, 64],
        help="Floating-point precision for the search.",
    )
    parser.add_argument(
        "--model-selection",
        default="best",
        choices=["best", "accuracy", "score"],
        help="How PySR chooses the final equation from its Pareto front.",
    )
    parser.add_argument(
        "--binary-operators",
        nargs="+",
        default=list(DEFAULT_BINARY_OPERATORS),
        help="Binary operators for PySR.",
    )
    parser.add_argument(
        "--unary-operators",
        nargs="+",
        default=list(DEFAULT_UNARY_OPERATORS),
        help="Unary operators for PySR.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed passed to PySR.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable PySR denoising.",
    )
    parser.add_argument(
        "--select-k-features",
        type=int,
        help="Ask PySR to preselect this many features before the search.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the dataset and print config without importing or running PySR.",
    )
    parser.add_argument(
        "--curriculum-rows",
        nargs="+",
        help=(
            "Optional staged row counts for curriculum training, for example "
            "`--curriculum-rows 1000 10000 100000 full`."
        ),
    )
    return parser.parse_args()


def _normalize_target_column(raw_index: int, n_columns: int) -> int:
    index = raw_index if raw_index >= 0 else n_columns + raw_index
    if index < 0 or index >= n_columns:
        raise ValueError(
            f"Target column index {raw_index} is out of range for {n_columns} columns."
        )
    return index


def _parse_numeric_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    expected_width: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = [piece for piece in SPLIT_PATTERN.split(stripped) if piece]
            row = [float(piece) for piece in parts]

            if expected_width is None:
                expected_width = len(row)
            elif len(row) != expected_width:
                raise ValueError(
                    f"Inconsistent column count in {path} on line {line_number}: "
                    f"expected {expected_width}, found {len(row)}."
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"No numeric rows found in {path}.")

    return np.asarray(rows, dtype=np.float64)


def load_aifeynman_dataset(
    path: str | Path,
    target_column: int = -1,
    vars_name: list[str] | None = None,
) -> DatasetBundle:
    dataset_path = Path(path).expanduser().resolve()
    data = _parse_numeric_table(dataset_path)
    n_rows, n_columns = data.shape

    if n_columns < 2:
        raise ValueError("Expected at least one feature column and one target column.")

    target_index = _normalize_target_column(target_column, n_columns)
    if vars_name:
        all_names = vars_name
    else:
        all_names = []
        feature_index = 0
        for column_index in range(n_columns):
            if column_index == target_index:
                all_names.append("y")
            else:
                all_names.append(f"x{feature_index}")
                feature_index += 1

    if len(all_names) != n_columns:
        raise ValueError(
            "--vars-name must provide exactly one name for every column in the dataset."
        )

    target_name = all_names[target_index]
    feature_names = [name for idx, name in enumerate(all_names) if idx != target_index]

    X = np.delete(data, target_index, axis=1)
    y = data[:, target_index]

    return DatasetBundle(
        path=dataset_path,
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        n_rows=n_rows,
        n_columns=n_columns,
    )


def build_model(args: argparse.Namespace, warm_start: bool = False):
    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise SystemExit(
            "PySR is not installed. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`, then rerun. "
            "On first import, PySR will also set up its Julia dependencies."
        ) from exc

    procs = None if args.procs == 0 else args.procs

    return PySRRegressor(
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        binary_operators=args.binary_operators,
        unary_operators=args.unary_operators,
        model_selection=args.model_selection,
        precision=args.precision,
        random_state=args.random_state,
        procs=procs,
        denoise=args.denoise,
        select_k_features=args.select_k_features,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        warm_start=warm_start,
    )


def print_dataset_summary(dataset: DatasetBundle) -> None:
    print(f"dataset={dataset.path}")
    print(f"rows={dataset.n_rows} columns={dataset.n_columns}")
    print(f"features={dataset.feature_names}")
    print(f"target={dataset.target_name}")


def _parse_stage_row_count(raw_value: str, total_rows: int) -> tuple[str, int]:
    normalized = raw_value.strip().lower()
    if normalized in {"full", "all", "max"}:
        return ("full", total_rows)
    try:
        row_count = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid curriculum stage '{raw_value}'. Use integers or 'full'."
        ) from exc
    if row_count < 1:
        raise ValueError("Curriculum row counts must be positive integers.")
    return (str(row_count), min(row_count, total_rows))


def build_curriculum_stages(
    dataset: DatasetBundle,
    curriculum_rows: list[str] | None,
    random_state: int,
) -> list[CurriculumStage]:
    if not curriculum_rows:
        return [
            CurriculumStage(
                index=1,
                label="full",
                row_count=dataset.n_rows,
                X=dataset.X,
                y=dataset.y,
            )
        ]

    parsed_rows = [
        _parse_stage_row_count(raw_value, dataset.n_rows) for raw_value in curriculum_rows
    ]
    parsed_rows.sort(key=lambda item: item[1])

    deduped: list[tuple[str, int]] = []
    for label, row_count in parsed_rows:
        if deduped and deduped[-1][1] == row_count:
            deduped[-1] = (label, row_count)
        else:
            deduped.append((label, row_count))

    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(dataset.n_rows)
    stages: list[CurriculumStage] = []
    for index, (label, row_count) in enumerate(deduped, start=1):
        if row_count >= dataset.n_rows:
            X_stage = dataset.X
            y_stage = dataset.y
            label_value = "full"
        else:
            stage_indices = permutation[:row_count]
            X_stage = dataset.X[stage_indices]
            y_stage = dataset.y[stage_indices]
            label_value = label
        stages.append(
            CurriculumStage(
                index=index,
                label=label_value,
                row_count=row_count,
                X=X_stage,
                y=y_stage,
            )
        )
    return stages


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(X)).reshape(-1)
    residuals = predictions - y
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    return {"mse": mse, "mae": mae}


def save_stage_results_csv(
    output_dir: Path,
    run_id: str,
    stage_results: list[RunStageResult],
) -> str:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "stage_results.csv"
    fieldnames = list(RunStageResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stage_results:
            writer.writerow(row.__dict__)
    return str(csv_path)


def _extract_best_equation_stats(model: Any) -> tuple[float | None, float | None, float | None]:
    equations = getattr(model, "equations_", None)
    if equations is None or len(equations) == 0:
        return (None, None, None)

    row = equations.iloc[-1]
    complexity = float(row["complexity"]) if "complexity" in row else None
    loss = float(row["loss"]) if "loss" in row else None
    score = float(row["score"]) if "score" in row else None
    return (complexity, loss, score)


def run_single_fit(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> None:
    model.fit(X, y, variable_names=feature_names)


def run_curriculum_fit(
    args: argparse.Namespace,
    dataset: DatasetBundle,
    output_dir: Path,
    run_id: str,
) -> list[RunStageResult]:
    stages = build_curriculum_stages(dataset, args.curriculum_rows, args.random_state)
    if len(stages) == 1:
        stage = stages[0]
        model = build_model(args)
        start_time = time.perf_counter()
        run_single_fit(model, stage.X, stage.y, dataset.feature_names)
        runtime_sec = time.perf_counter() - start_time
        stage_eval = evaluate_model(model, stage.X, stage.y)
        full_eval = evaluate_model(model, dataset.X, dataset.y)
        complexity, loss, score = _extract_best_equation_stats(model)
        print("\nbest_equation:")
        print(model.sympy())
        print("\nmetrics:")
        print(f"runtime_sec={runtime_sec:.3f}")
        print(f"stage_mse={stage_eval['mse']:.6e}")
        print(f"stage_mae={stage_eval['mae']:.6e}")
        print(f"full_mse={full_eval['mse']:.6e}")
        print(f"full_mae={full_eval['mae']:.6e}")
        stage_name = "full" if stage.row_count == dataset.n_rows else f"stage_01_{stage.row_count}rows"
        stage_result = RunStageResult(
            run_id=run_id,
            dataset_name=dataset.path.name,
            dataset_path=str(dataset.path),
            seed=args.random_state,
            stage_index=stage.index,
            stage_name=stage_name,
            stage_label=stage.label,
            stage_rows=stage.row_count,
            total_rows=dataset.n_rows,
            best_equation=str(model.sympy()),
            latex=str(model.latex()),
            runtime_sec=runtime_sec,
            stage_mse=stage_eval["mse"],
            stage_mae=stage_eval["mae"],
            full_mse=full_eval["mse"],
            full_mae=full_eval["mae"],
            complexity=complexity,
            loss=loss,
            score=score,
        )
        csv_path = save_stage_results_csv(output_dir, run_id, [stage_result])
        print(f"\nstage_results_csv={csv_path}")
        return [stage_result]

    model = build_model(args, warm_start=True)
    stage_results: list[RunStageResult] = []

    for stage in stages:
        stage_name = f"stage_{stage.index:02d}_{stage.row_count}rows"
        print(
            f"\n=== curriculum stage {stage.index}/{len(stages)}: "
            f"{stage.row_count} rows ==="
        )
        start_time = time.perf_counter()
        run_single_fit(model, stage.X, stage.y, dataset.feature_names)
        runtime_sec = time.perf_counter() - start_time
        stage_eval = evaluate_model(model, stage.X, stage.y)
        full_eval = evaluate_model(model, dataset.X, dataset.y)
        complexity, loss, score = _extract_best_equation_stats(model)

        print("best_equation:")
        print(model.sympy())
        print("metrics:")
        print(f"runtime_sec={runtime_sec:.3f}")
        print(f"stage_mse={stage_eval['mse']:.6e}")
        print(f"stage_mae={stage_eval['mae']:.6e}")
        print(f"full_mse={full_eval['mse']:.6e}")
        print(f"full_mae={full_eval['mae']:.6e}")
        stage_results.append(
            RunStageResult(
                run_id=run_id,
                dataset_name=dataset.path.name,
                dataset_path=str(dataset.path),
                seed=args.random_state,
                stage_index=stage.index,
                stage_name=stage_name,
                stage_label=stage.label,
                stage_rows=stage.row_count,
                total_rows=dataset.n_rows,
                best_equation=str(model.sympy()),
                latex=str(model.latex()),
                runtime_sec=runtime_sec,
                stage_mse=stage_eval["mse"],
                stage_mae=stage_eval["mae"],
                full_mse=full_eval["mse"],
                full_mae=full_eval["mae"],
                complexity=complexity,
                loss=loss,
                score=score,
            )
        )

    csv_path = save_stage_results_csv(output_dir, run_id, stage_results)
    print(f"\nstage_results_csv={csv_path}")
    return stage_results


def run_from_args(args: argparse.Namespace) -> list[RunStageResult]:
    dataset = load_aifeynman_dataset(
        args.data,
        target_column=args.target_column,
        vars_name=args.vars_name,
    )
    print_dataset_summary(dataset)

    stages = build_curriculum_stages(dataset, args.curriculum_rows, args.random_state)
    if len(stages) > 1:
        print("curriculum_stages=" + ", ".join(str(stage.row_count) for stage in stages))

    if args.dry_run:
        print("dry_run=true")
        return []

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or dataset.path.stem
    return run_curriculum_fit(args, dataset, output_dir, run_id)


def main() -> None:
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
