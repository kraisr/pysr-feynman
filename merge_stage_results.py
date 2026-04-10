#!/usr/bin/env python3
"""Merge per-run stage_results.csv files into one aggregate CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all stage_results.csv files under an output directory.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-run output folders.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional destination CSV path. Defaults to <input-dir>/benchmark_results_merged.csv.",
    )
    return parser.parse_args()


def load_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*/stage_results.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    return rows


def write_rows(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No stage_results.csv files found for {output_csv.parent}.")
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else input_dir / "benchmark_results_merged.csv"
    )
    rows = load_rows(input_dir)
    write_rows(rows, output_csv)
    print(f"merged_csv={output_csv}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
