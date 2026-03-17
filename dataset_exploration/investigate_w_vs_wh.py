#!/usr/bin/env python3
"""Investigate differences between `_W` and `_Wh` Ireland data files.

The script pairs files such as ``H1_W.csv`` and ``H1_Wh.csv``, then reports:
- Schema differences.
- Timestamp coverage differences.
- Numeric consistency for comparable columns where ``W / 60`` is expected to
  match ``Wh`` at the same timestamp.

Usage example::

    python dataset_exploration/investigate_w_vs_wh.py \
        --data-dir dataset_exploration/ireland_data
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

FILE_PATTERN = re.compile(r"^H(?P<house>\d+)_(?P<unit>W|Wh)\.csv$")
DT_FORMAT = "%Y-%m-%d %H:%M:%S"

W_TO_WH_FIELDS = {
    "Discharge(W)": "Discharge(Wh)",
    "Charge(W)": "Charge(Wh)",
    "Production(W)": "Production(Wh)",
    "Consumption(W)": "Consumption(Wh)",
}

DIRECT_COMPARE_FIELDS = {
    "State of Charge(%)": "State of Charge(%)",
}


@dataclass
class RunningError:
    """Accumulate error metrics for a single compared feature.

    :param count: Number of compared rows.
    :param abs_error_sum: Sum of absolute errors.
    :param sq_error_sum: Sum of squared errors.
    :param signed_error_sum: Sum of signed errors.
    :param max_abs_error: Maximum absolute error.
    :param within_tolerance: Count of rows with absolute error <= tolerance.
    """

    count: int = 0
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0
    signed_error_sum: float = 0.0
    max_abs_error: float = 0.0
    within_tolerance: int = 0

    def update(self, expected: float, actual: float, tolerance: float) -> None:
        """Update metrics with one observation.

        :param expected: Expected value.
        :param actual: Observed value.
        :param tolerance: Non-negative absolute tolerance.
        """
        error = actual - expected
        abs_error = abs(error)
        self.count += 1
        self.abs_error_sum += abs_error
        self.sq_error_sum += error * error
        self.signed_error_sum += error
        self.max_abs_error = max(self.max_abs_error, abs_error)
        if abs_error <= tolerance:
            self.within_tolerance += 1

    def to_display(self) -> str:
        """Format aggregate metrics for terminal output.

        :return: Human-readable summary string.
        """
        if self.count == 0:
            return "n=0"
        mae = self.abs_error_sum / self.count
        rmse = math.sqrt(self.sq_error_sum / self.count)
        mean_error = self.signed_error_sum / self.count
        pct_within = 100.0 * self.within_tolerance / self.count
        return (
            f"n={self.count:,}, mae={mae:.6f}, rmse={rmse:.6f}, "
            f"mean_error={mean_error:.6f}, max_abs={self.max_abs_error:.6f}, "
            f"within_tol={pct_within:.2f}%"
        )


@dataclass
class TableStats:
    """Statistics collected while reading a CSV file.

    :param row_count: Number of data rows.
    :param bad_timestamp_rows: Rows where ``date`` could not be parsed.
    :param duplicate_timestamps: Number of repeated timestamps.
    :param min_timestamp: Earliest parsed timestamp.
    :param max_timestamp: Latest parsed timestamp.
    :param columns: Header names after stripping whitespace.
    """

    row_count: int = 0
    bad_timestamp_rows: int = 0
    duplicate_timestamps: int = 0
    min_timestamp: Optional[datetime] = None
    max_timestamp: Optional[datetime] = None
    columns: Sequence[str] = ()


def parse_float(value: str) -> Optional[float]:
    """Parse a CSV scalar value into float.

    :param value: Raw field string.
    :return: Parsed float, or ``None`` for empty/non-numeric data.
    """
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def strip_keys(row: Mapping[str, str]) -> Dict[str, str]:
    """Normalize dictionary keys by trimming whitespace.

    :param row: Original row mapping.
    :return: Key-normalized dictionary.
    """
    return {str(key).strip(): value for key, value in row.items()}


def read_table(
    csv_path: Path, expected_fields: Iterable[str]
) -> Tuple[TableStats, Dict[str, Dict[str, Optional[float]]], Set[str]]:
    """Read one CSV into timestamp-keyed numeric values and profile statistics.

    :param csv_path: Path to the CSV file.
    :param expected_fields: Fields to parse as floats.
    :return: Tuple of stats, per-timestamp numeric map, and timestamp string set.
    """
    values_by_timestamp: Dict[str, Dict[str, Optional[float]]] = {}
    seen_timestamps: Set[str] = set()
    expected_set = set(expected_fields)
    stats = TableStats()

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            return stats, values_by_timestamp, seen_timestamps

        stats.columns = tuple(name.strip() for name in reader.fieldnames)

        for raw_row in reader:
            row = strip_keys(raw_row)
            stats.row_count += 1

            timestamp = (row.get("date") or "").strip()
            try:
                parsed = datetime.strptime(timestamp, DT_FORMAT)
            except ValueError:
                stats.bad_timestamp_rows += 1
                continue

            if stats.min_timestamp is None or parsed < stats.min_timestamp:
                stats.min_timestamp = parsed
            if stats.max_timestamp is None or parsed > stats.max_timestamp:
                stats.max_timestamp = parsed

            if timestamp in seen_timestamps:
                stats.duplicate_timestamps += 1
            seen_timestamps.add(timestamp)

            values_by_timestamp[timestamp] = {
                field: parse_float(row.get(field, ""))
                for field in expected_set
            }

    return stats, values_by_timestamp, seen_timestamps


def format_dt(value: Optional[datetime]) -> str:
    """Format datetime for display.

    :param value: Datetime to format.
    :return: Timestamp string or ``NA``.
    """
    if value is None:
        return "NA"
    return value.strftime(DT_FORMAT)


def discover_house_pairs(data_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Discover matching ``_W`` and ``_Wh`` files by house id.

    :param data_dir: Directory containing CSV files.
    :return: Mapping ``house_id -> {'W': path, 'Wh': path}``.
    """
    grouped: Dict[str, Dict[str, Path]] = {}
    for file_path in sorted(data_dir.glob("*.csv")):
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            continue
        house = match.group("house")
        unit = match.group("unit")
        grouped.setdefault(house, {})[unit] = file_path
    return grouped


def compare_house(w_path: Path, wh_path: Path, tolerance: float) -> Tuple[List[str], Dict[str, RunningError]]:
    """Compare one ``_W`` file and one ``_Wh`` file.

    :param w_path: Path to power file.
    :param wh_path: Path to energy file.
    :param tolerance: Absolute tolerance for error checks.
    :return: Human-readable lines and metric aggregates by compared feature.
    """
    compare_metrics: Dict[str, RunningError] = {}
    for left, right in W_TO_WH_FIELDS.items():
        compare_metrics[f"{left} -> {right} (W/60)"] = RunningError()
    for left, right in DIRECT_COMPARE_FIELDS.items():
        compare_metrics[f"{left} -> {right}"] = RunningError()

    required_w_fields = list(W_TO_WH_FIELDS) + list(DIRECT_COMPARE_FIELDS)
    required_wh_fields = list(W_TO_WH_FIELDS.values()) + list(DIRECT_COMPARE_FIELDS.values())

    w_stats, w_values, w_timestamps = read_table(w_path, required_w_fields)
    wh_stats, wh_values, wh_timestamps = read_table(wh_path, required_wh_fields)

    common_timestamps = w_timestamps & wh_timestamps
    w_only_timestamps = w_timestamps - wh_timestamps
    wh_only_timestamps = wh_timestamps - w_timestamps

    lines: List[str] = []
    lines.append(f"W rows={w_stats.row_count:,}, Wh rows={wh_stats.row_count:,}")
    lines.append(
        "W time range={} -> {} | Wh time range={} -> {}".format(
            format_dt(w_stats.min_timestamp),
            format_dt(w_stats.max_timestamp),
            format_dt(wh_stats.min_timestamp),
            format_dt(wh_stats.max_timestamp),
        )
    )
    lines.append(
        "timestamps: common={:,}, W_only={:,}, Wh_only={:,}".format(
            len(common_timestamps), len(w_only_timestamps), len(wh_only_timestamps)
        )
    )
    lines.append(
        f"bad timestamps: W={w_stats.bad_timestamp_rows}, Wh={wh_stats.bad_timestamp_rows}; "
        f"duplicates: W={w_stats.duplicate_timestamps}, Wh={wh_stats.duplicate_timestamps}"
    )

    w_columns = set(w_stats.columns)
    wh_columns = set(wh_stats.columns)
    lines.append(f"W-only columns: {sorted(w_columns - wh_columns)}")
    lines.append(f"Wh-only columns: {sorted(wh_columns - w_columns)}")

    for timestamp in common_timestamps:
        w_row = w_values[timestamp]
        wh_row = wh_values[timestamp]

        for w_field, wh_field in W_TO_WH_FIELDS.items():
            w_val = w_row.get(w_field)
            wh_val = wh_row.get(wh_field)
            if w_val is None or wh_val is None:
                continue
            expected_wh = w_val / 60.0
            key = f"{w_field} -> {wh_field} (W/60)"
            compare_metrics[key].update(expected_wh, wh_val, tolerance)

        for w_field, wh_field in DIRECT_COMPARE_FIELDS.items():
            w_val = w_row.get(w_field)
            wh_val = wh_row.get(wh_field)
            if w_val is None or wh_val is None:
                continue
            key = f"{w_field} -> {wh_field}"
            compare_metrics[key].update(w_val, wh_val, tolerance)

    return lines, compare_metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return: Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Investigate differences between paired _W and _Wh files."
    )
    parser.add_argument(
        "--data-dir",
        default="dataset_exploration/ireland_data",
        help="Directory containing H*_W.csv and H*_Wh.csv files.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Absolute tolerance used for within-tolerance summary (default: 0.05).",
    )
    return parser.parse_args()


def main() -> int:
    """Run CLI entrypoint.

    :return: Exit status code.
    """
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}")
        return 2

    house_files = discover_house_pairs(data_dir)
    paired_houses = sorted(
        house for house, units in house_files.items() if "W" in units and "Wh" in units
    )
    if not paired_houses:
        print(f"No paired _W/_Wh files found under {data_dir}")
        return 1

    overall_metrics: Dict[str, RunningError] = {}

    print(f"Found {len(paired_houses)} paired houses in {data_dir}")
    print(f"Tolerance: {args.tolerance}\n")

    for house in paired_houses:
        w_path = house_files[house]["W"]
        wh_path = house_files[house]["Wh"]

        print(f"=== House H{house} ===")
        lines, house_metrics = compare_house(w_path, wh_path, args.tolerance)
        for line in lines:
            print(line)

        for metric_name, metric in house_metrics.items():
            print(f"{metric_name}: {metric.to_display()}")
            agg = overall_metrics.setdefault(metric_name, RunningError())
            agg.count += metric.count
            agg.abs_error_sum += metric.abs_error_sum
            agg.sq_error_sum += metric.sq_error_sum
            agg.signed_error_sum += metric.signed_error_sum
            agg.max_abs_error = max(agg.max_abs_error, metric.max_abs_error)
            agg.within_tolerance += metric.within_tolerance

        print("")

    print("=== Overall Summary ===")
    for metric_name, metric in overall_metrics.items():
        print(f"{metric_name}: {metric.to_display()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
