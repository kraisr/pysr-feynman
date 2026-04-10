#!/usr/bin/env python3
"""Create direct baseline-vs-curriculum comparison plots."""

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
    parser = argparse.ArgumentParser(
        description="Compare baseline and curriculum benchmark results side by side.",
    )
    parser.add_argument(
        "--baseline-csv",
        required=True,
        help="Path to the merged baseline benchmark CSV.",
    )
    parser.add_argument(
        "--curriculum-csv",
        required=True,
        help="Path to the merged curriculum benchmark CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/comparison",
        help="Directory to save comparison plots and summary table.",
    )
    return parser.parse_args()


def select_final_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    ordered = dataframe.sort_values(["dataset_name", "seed", "stage_rows"])
    final_rows = ordered.groupby(["dataset_name", "seed"], as_index=False).tail(1).copy()
    final_rows = final_rows.sort_values(["dataset_name", "seed"]).reset_index(drop=True)
    return final_rows


def build_comparison_table(
    baseline_final: pd.DataFrame,
    curriculum_final: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_final.rename(
        columns={
            "run_id": "baseline_run_id",
            "runtime_sec": "baseline_runtime_sec",
            "full_mse": "baseline_full_mse",
            "full_mae": "baseline_full_mae",
            "best_equation": "baseline_equation",
            "complexity": "baseline_complexity",
        }
    )
    curriculum = curriculum_final.rename(
        columns={
            "run_id": "curriculum_run_id",
            "runtime_sec": "curriculum_runtime_sec",
            "full_mse": "curriculum_full_mse",
            "full_mae": "curriculum_full_mae",
            "best_equation": "curriculum_equation",
            "complexity": "curriculum_complexity",
        }
    )
    keep_baseline = [
        "dataset_name",
        "seed",
        "baseline_run_id",
        "baseline_runtime_sec",
        "baseline_full_mse",
        "baseline_full_mae",
        "baseline_equation",
        "baseline_complexity",
    ]
    keep_curriculum = [
        "dataset_name",
        "seed",
        "curriculum_run_id",
        "curriculum_runtime_sec",
        "curriculum_full_mse",
        "curriculum_full_mae",
        "curriculum_equation",
        "curriculum_complexity",
    ]
    comparison = baseline[keep_baseline].merge(
        curriculum[keep_curriculum],
        on=["dataset_name", "seed"],
        how="outer",
        validate="one_to_one",
    )
    comparison["runtime_speedup"] = (
        comparison["baseline_runtime_sec"] / comparison["curriculum_runtime_sec"]
    )
    comparison["mse_ratio"] = (
        comparison["baseline_full_mse"] / comparison["curriculum_full_mse"]
    )
    comparison["better_mse"] = comparison.apply(
        lambda row: "baseline"
        if row["baseline_full_mse"] < row["curriculum_full_mse"]
        else "curriculum",
        axis=1,
    )
    comparison["faster_method"] = comparison.apply(
        lambda row: "baseline"
        if row["baseline_runtime_sec"] < row["curriculum_runtime_sec"]
        else "curriculum",
        axis=1,
    )
    return comparison.sort_values(["dataset_name", "seed"]).reset_index(drop=True)


def plot_grouped_bars(
    comparison: pd.DataFrame,
    output_path: Path,
    left_column: str,
    right_column: str,
    y_label: str,
    title: str,
    log_scale: bool = False,
) -> None:
    labels = comparison["dataset_name"].astype(str).tolist()
    positions = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 6))
    ax.bar(
        [position - width / 2 for position in positions],
        comparison[left_column],
        width=width,
        label="Baseline",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        comparison[right_column],
        width=width,
        label="Curriculum",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_summary_table(comparison: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "comparison_summary.csv"
    comparison.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_csv).expanduser().resolve()
    curriculum_path = Path(args.curriculum_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(baseline_path)
    curriculum_df = pd.read_csv(curriculum_path)
    if baseline_df.empty or curriculum_df.empty:
        raise SystemExit("Baseline or curriculum CSV is empty.")

    baseline_final = select_final_rows(baseline_df)
    curriculum_final = select_final_rows(curriculum_df)
    comparison = build_comparison_table(baseline_final, curriculum_final)

    plot_grouped_bars(
        comparison,
        output_dir / "final_mse_comparison.png",
        "baseline_full_mse",
        "curriculum_full_mse",
        "Final full-dataset MSE",
        "Baseline vs Curriculum: Final Error",
        log_scale=True,
    )
    plot_grouped_bars(
        comparison,
        output_dir / "runtime_comparison.png",
        "baseline_runtime_sec",
        "curriculum_runtime_sec",
        "Runtime (seconds)",
        "Baseline vs Curriculum: Runtime",
        log_scale=False,
    )

    summary_path = save_summary_table(comparison, output_dir)
    print(f"comparison_summary={summary_path}")
    print(f"plots_saved={output_dir}")


if __name__ == "__main__":
    main()
