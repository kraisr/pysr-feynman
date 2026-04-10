#!/usr/bin/env python3
"""Create a few quick plots from benchmark_results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Missing plotting dependencies. Install the project environment from "
        "`environment.yml` and rerun."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PySR benchmark outputs.")
    parser.add_argument(
        "--results-csv",
        default="outputs/benchmark_results.csv",
        help="Path to the aggregate benchmark CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save generated plots.",
    )
    return parser.parse_args()


def plot_runtime_vs_error(dataframe: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dataframe["runtime_sec"], dataframe["full_mse"], alpha=0.8)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Full-dataset MSE")
    ax.set_yscale("log")
    ax.set_title("Runtime vs Full-Dataset Error")
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_full_mse.png", dpi=200)
    plt.close(fig)


def plot_curriculum_progress(dataframe: pd.DataFrame, output_dir: Path) -> None:
    curriculum = dataframe.sort_values(["dataset_name", "seed", "stage_rows"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for (dataset_name, seed), group in curriculum.groupby(["dataset_name", "seed"]):
        ax.plot(
            group["stage_rows"],
            group["full_mse"],
            marker="o",
            alpha=0.8,
            label=f"{dataset_name} seed={seed}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Stage rows")
    ax.set_ylabel("Full-dataset MSE")
    ax.set_title("Curriculum Stage Size vs Error")
    if len(curriculum.groupby(["dataset_name", "seed"])) <= 10:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "curriculum_stage_vs_full_mse.png", dpi=200)
    plt.close(fig)


def plot_final_stage_boxplot(dataframe: pd.DataFrame, output_dir: Path) -> None:
    final_rows = dataframe.sort_values("stage_rows").groupby(["dataset_name", "seed"]).tail(1)
    grouped = final_rows.groupby("dataset_name")["full_mse"]
    labels = list(grouped.groups.keys())
    values = [grouped.get_group(label).values for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    ax.boxplot(values, labels=labels)
    ax.set_yscale("log")
    ax.set_ylabel("Final full-dataset MSE")
    ax.set_title("Final-Stage Error Across Seeds")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "final_stage_boxplot.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(results_path)
    if dataframe.empty:
        raise SystemExit(f"No rows found in {results_path}.")

    plot_runtime_vs_error(dataframe, output_dir)
    plot_curriculum_progress(dataframe, output_dir)
    plot_final_stage_boxplot(dataframe, output_dir)
    print(f"plots_saved={output_dir}")


if __name__ == "__main__":
    main()
