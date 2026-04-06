#!/usr/bin/env python3
"""Build a normalized, time-aligned grid dataset from raw demand/frequency/generation files."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class InterpolationSeries:
    """Numeric interpolation data for one feature column."""

    seconds: np.ndarray
    values: np.ndarray


_WORKER_INTERPOLATION_TABLES: OrderedDict[str, InterpolationSeries] | None = None
_WORKER_OUTPUT_DIR: Path | None = None
_WORKER_CHUNKSIZE: int | None = None
_WORKER_GZIP_LEVEL: int | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Emit normalized frequency-aligned files by interpolating demand and generation "
            "data at each frequency timestamp."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("grid_energy_data/dataset"),
        help="Path containing demand/, frequency/, and df_fuel_ckan.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grid_energy_data/normalized"),
        help="Directory where normalized partition files will be written.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Chunk size used while streaming frequency files.",
    )
    parser.add_argument(
        "--demand-chunksize",
        type=int,
        default=200_000,
        help="Chunk size used while loading annual demand files.",
    )
    parser.add_argument(
        "--generation-chunksize",
        type=int,
        default=200_000,
        help="Chunk size used while loading generation mix file.",
    )
    parser.add_argument(
        "--include-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include interpolated generation-mix columns in normalized output.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite already existing normalized output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel workers for frequency-file normalization.",
    )
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Parallel executor type used for frequency-file normalization.",
    )
    parser.add_argument(
        "--gzip-compresslevel",
        type=int,
        default=9,
        help="Gzip compression level for normalized output (0-9).",
    )
    return parser.parse_args()


def sanitize_name(column_name: str) -> str:
    """Normalize a column name into a lowercase snake-style token.

    :param column_name: Original column name.
    :return: Sanitized name suitable for output column labels.
    """
    lowered = column_name.strip().lower()
    lowered = lowered.replace("%", "pct")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered


def parse_frequency_timestamps(raw_values: pd.Series) -> pd.Series:
    """Parse mixed-format frequency timestamps into UTC timestamps.

    :param raw_values: Raw `dtm` string column from frequency files.
    :return: Parsed UTC timestamp series, with unparsable values as `NaT`.
    """
    values = raw_values.astype("string").str.strip()
    parsed = pd.to_datetime(
        values,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
        utc=True,
    )

    unresolved = parsed.isna()
    if unresolved.any():
        parsed.loc[unresolved] = pd.to_datetime(
            values.loc[unresolved],
            format="%Y-%m-%d %H:%M:%S %z",
            errors="coerce",
            utc=True,
        )

    unresolved = parsed.isna()
    if unresolved.any():
        parsed.loc[unresolved] = pd.to_datetime(
            values.loc[unresolved],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
            utc=True,
        )

    return parsed


def discover_demand_columns(demand_files: Iterable[Path]) -> list[str]:
    """Collect the union of demand columns while preserving first-seen order.

    :param demand_files: Sequence of annual demand CSV files.
    :return: Ordered list of all numeric demand feature columns.
    """
    ordered: OrderedDict[str, None] = OrderedDict()
    ignored = {"SETTLEMENT_DATE", "SETTLEMENT_PERIOD"}
    for path in demand_files:
        header = pd.read_csv(path, nrows=0).columns
        for column in header:
            cleaned = column.strip()
            if cleaned in ignored:
                continue
            ordered.setdefault(cleaned, None)
    return list(ordered.keys())


def build_demand_anchor_table(
    demand_files: list[Path],
    demand_columns: list[str],
    chunksize: int,
) -> pd.DataFrame:
    """Load and normalize demand anchor points.

    :param demand_files: Annual demand CSV files.
    :param demand_columns: Union of demand feature columns.
    :param chunksize: Streaming chunk size.
    :return: Dataframe with `timestamp_utc` and numeric demand columns.
    """
    # pylint: disable=too-many-locals
    frames: list[pd.DataFrame] = []

    for path in tqdm(demand_files, desc="Loading demand"):
        reader = pd.read_csv(path, dtype="string", low_memory=False, chunksize=chunksize)
        for chunk in reader:
            chunk.columns = [column.strip() for column in chunk.columns]
            for column in demand_columns:
                if column not in chunk.columns:
                    chunk[column] = pd.NA

            settlement_date = pd.to_datetime(
                chunk["SETTLEMENT_DATE"].astype("string").str.strip(),
                format="%Y-%m-%d",
                errors="coerce",
                utc=True,
            )
            settlement_period = pd.to_numeric(
                chunk["SETTLEMENT_PERIOD"].astype("string").str.strip(),
                errors="coerce",
            )

            offset_minutes = (settlement_period - 1.0) * 30.0
            timestamp = settlement_date + pd.to_timedelta(offset_minutes, unit="m")
            valid = timestamp.notna()
            if not valid.any():
                continue

            numeric_data = {}
            for column in demand_columns:
                numeric_data[column] = pd.to_numeric(
                    chunk.loc[valid, column].astype("string").str.strip(),
                    errors="coerce",
                ).to_numpy(dtype=np.float64)

            frame = pd.DataFrame(numeric_data)
            frame.insert(0, "timestamp_utc", timestamp.loc[valid].to_numpy())
            frames.append(frame)

    if not frames:
        raise RuntimeError("No demand anchor points were loaded.")

    demand = pd.concat(frames, ignore_index=True)
    demand = demand.sort_values("timestamp_utc")
    demand = demand.groupby("timestamp_utc", as_index=False).mean(numeric_only=True)
    return demand


def build_generation_anchor_table(path: Path, chunksize: int) -> pd.DataFrame:
    """Load and normalize generation-mix anchor points.

    :param path: Generation mix CSV path.
    :param chunksize: Streaming chunk size.
    :return: Dataframe with `timestamp_utc` and numeric generation columns.
    """
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(path, dtype="string", low_memory=False, chunksize=chunksize)
    for chunk in tqdm(reader, desc="Loading generation"):
        chunk.columns = [column.strip() for column in chunk.columns]
        if "DATETIME" not in chunk.columns:
            continue

        timestamp = pd.to_datetime(
            chunk["DATETIME"].astype("string").str.strip(),
            errors="coerce",
            utc=True,
        )
        valid = timestamp.notna()
        if not valid.any():
            continue

        numeric_columns = [column for column in chunk.columns if column != "DATETIME"]
        numeric_data = {}
        for column in numeric_columns:
            numeric_data[column] = pd.to_numeric(
                chunk.loc[valid, column].astype("string").str.strip(),
                errors="coerce",
            ).to_numpy(dtype=np.float64)

        frame = pd.DataFrame(numeric_data)
        frame.insert(0, "timestamp_utc", timestamp.loc[valid].to_numpy())
        frames.append(frame)

    if not frames:
        raise RuntimeError("No generation anchor points were loaded.")

    generation = pd.concat(frames, ignore_index=True)
    generation = generation.sort_values("timestamp_utc")
    generation = generation.groupby("timestamp_utc", as_index=False).mean(numeric_only=True)
    return generation


def build_interpolation_tables(
    frame: pd.DataFrame,
    prefix: str,
) -> OrderedDict[str, InterpolationSeries]:
    """Build interpolation arrays for every numeric feature in a frame.

    :param frame: Anchor dataframe with `timestamp_utc` plus numeric features.
    :param prefix: Output column prefix, for example `demand_`.
    :return: Ordered mapping from output column names to interpolation series.
    """
    tables: OrderedDict[str, InterpolationSeries] = OrderedDict()
    seconds = frame["timestamp_utc"].astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000

    for column in frame.columns:
        if column == "timestamp_utc":
            continue

        values = frame[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            continue

        series_seconds = seconds[valid]
        series_values = values[valid]
        unique_mask = np.concatenate(([True], np.diff(series_seconds) > 0))
        series_seconds = series_seconds[unique_mask]
        series_values = series_values[unique_mask]
        if len(series_seconds) < 2:
            continue

        output_name = f"{prefix}{sanitize_name(column)}"
        tables[output_name] = InterpolationSeries(
            seconds=series_seconds.astype(np.float64),
            values=series_values.astype(np.float64),
        )

    return tables


def interpolate_chunk_columns(
    timestamp_seconds: np.ndarray,
    tables: OrderedDict[str, InterpolationSeries],
) -> dict[str, np.ndarray]:
    """Interpolate all configured columns at the provided timestamps.

    :param timestamp_seconds: Frequency timestamps in Unix seconds.
    :param tables: Mapping of interpolation tables.
    :return: Dictionary mapping output column names to interpolated arrays.
    """
    output: dict[str, np.ndarray] = {}
    for column_name, table in tables.items():
        output[column_name] = np.interp(
            timestamp_seconds,
            table.seconds,
            table.values,
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)
    return output


def first_frequency_timestamp(path: Path) -> pd.Timestamp:
    """Read the first timestamp present in one frequency file.

    :param path: Frequency CSV or ZIP path.
    :return: First valid UTC timestamp.
    """
    sample = pd.read_csv(
        path,
        dtype="string",
        low_memory=False,
        compression="infer",
        nrows=16,
    )
    sample.columns = [column.strip() for column in sample.columns]
    if "dtm" not in sample.columns:
        raise ValueError(f"Frequency file {path} does not contain a 'dtm' column.")

    parsed = parse_frequency_timestamps(sample["dtm"])
    valid = parsed.dropna()
    if valid.empty:
        raise ValueError(f"Frequency file {path} has no parseable timestamps in its head rows.")
    return valid.iloc[0]


def normalize_single_frequency_file(
    source_path: Path,
    interpolation_tables: OrderedDict[str, InterpolationSeries],
    output_dir: Path,
    chunksize: int,
    gzip_compresslevel: int,
) -> tuple[str, dict[str, object]]:
    """Normalize one frequency file and emit one normalized output file.

    :param source_path: Frequency CSV/ZIP input file.
    :param interpolation_tables: Combined interpolation tables for all added features.
    :param output_dir: Destination directory for normalized partitions.
    :param chunksize: CSV chunk size.
    :param gzip_compresslevel: Compression level for gzip output.
    :return: Tuple of source filename and one manifest entry.
    """
    # pylint: disable=too-many-locals
    destination = output_dir / f"{source_path.stem}_normalized.csv.gz"
    if destination.exists():
        destination.unlink()

    rows_written = 0
    min_timestamp = None
    max_timestamp = None
    header_pending = True

    reader = pd.read_csv(
        source_path,
        dtype="string",
        low_memory=False,
        compression="infer",
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk.columns = [column.strip() for column in chunk.columns]
        if "dtm" not in chunk.columns or "f" not in chunk.columns:
            continue

        timestamps = parse_frequency_timestamps(chunk["dtm"])
        frequency = pd.to_numeric(
            chunk["f"].astype("string").str.strip(),
            errors="coerce",
        )
        valid = timestamps.notna() & frequency.notna()
        if not valid.any():
            continue

        selected_timestamps = timestamps.loc[valid].reset_index(drop=True)
        selected_frequency = frequency.loc[valid].to_numpy(dtype=np.float32)
        timestamp_seconds = (
            selected_timestamps.astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000
        ).astype(np.float64)

        chunk_output: dict[str, np.ndarray] = {
            "timestamp_utc": selected_timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
            "frequency_hz": selected_frequency,
        }
        chunk_output.update(interpolate_chunk_columns(timestamp_seconds, interpolation_tables))

        normalized_chunk = pd.DataFrame(chunk_output)
        normalized_chunk.to_csv(
            destination,
            mode="w" if header_pending else "a",
            index=False,
            header=header_pending,
            compression={
                "method": "gzip",
                "compresslevel": gzip_compresslevel,
            },
        )

        header_pending = False
        rows_written += len(normalized_chunk)
        chunk_min = selected_timestamps.iloc[0]
        chunk_max = selected_timestamps.iloc[-1]
        min_timestamp = chunk_min if min_timestamp is None else min(min_timestamp, chunk_min)
        max_timestamp = chunk_max if max_timestamp is None else max(max_timestamp, chunk_max)

    return source_path.name, {
        "output_file": destination.name,
        "rows_written": rows_written,
        "min_timestamp_utc": None if min_timestamp is None else min_timestamp.isoformat(),
        "max_timestamp_utc": None if max_timestamp is None else max_timestamp.isoformat(),
    }


def init_frequency_worker(
    interpolation_tables: OrderedDict[str, InterpolationSeries],
    output_dir: Path,
    chunksize: int,
    gzip_compresslevel: int,
) -> None:
    """Initialize process-worker globals used by frequency normalization.

    :param interpolation_tables: Interpolation tables shared with worker.
    :param output_dir: Output directory for normalized files.
    :param chunksize: Read chunksize for frequency files.
    :param gzip_compresslevel: Compression level for output files.
    """
    # pylint: disable=global-statement
    global _WORKER_INTERPOLATION_TABLES
    global _WORKER_OUTPUT_DIR
    global _WORKER_CHUNKSIZE
    global _WORKER_GZIP_LEVEL

    _WORKER_INTERPOLATION_TABLES = interpolation_tables
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_CHUNKSIZE = chunksize
    _WORKER_GZIP_LEVEL = gzip_compresslevel


def normalize_single_frequency_file_worker(source_path: Path) -> tuple[str, dict[str, object]]:
    """Process-worker wrapper for one source frequency file.

    :param source_path: Frequency source file path.
    :return: Tuple of source filename and manifest entry.
    """
    if (
        _WORKER_INTERPOLATION_TABLES is None
        or _WORKER_OUTPUT_DIR is None
        or _WORKER_CHUNKSIZE is None
        or _WORKER_GZIP_LEVEL is None
    ):
        raise RuntimeError("Worker was not initialized before normalization.")

    return normalize_single_frequency_file(
        source_path=source_path,
        interpolation_tables=_WORKER_INTERPOLATION_TABLES,
        output_dir=_WORKER_OUTPUT_DIR,
        chunksize=_WORKER_CHUNKSIZE,
        gzip_compresslevel=_WORKER_GZIP_LEVEL,
    )


def normalize_frequency_files(
    frequency_files: list[Path],
    interpolation_tables: OrderedDict[str, InterpolationSeries],
    output_dir: Path,
    chunksize: int,
    overwrite: bool,
    workers: int,
    executor_type: str,
    gzip_compresslevel: int,
) -> dict[str, dict[str, object]]:
    """Normalize all frequency files and emit one normalized file per source file.

    :param frequency_files: Frequency CSV/ZIP files.
    :param interpolation_tables: Combined interpolation tables for all added features.
    :param output_dir: Destination directory for normalized partitions.
    :param chunksize: CSV chunk size.
    :param overwrite: Whether to overwrite existing outputs.
    :param workers: Number of parallel workers.
    :param executor_type: Either `thread` or `process`.
    :param gzip_compresslevel: Compression level for gzip outputs.
    :return: Per-file summary metadata for a manifest.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    output_dir.mkdir(parents=True, exist_ok=True)

    destinations = [
        output_dir / f"{source_path.stem}_normalized.csv.gz" for source_path in frequency_files
    ]
    if not overwrite:
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(
                f"{existing[0]} already exists. Re-run with --overwrite to replace it."
            )
    else:
        for path in destinations:
            if path.exists():
                path.unlink()

    manifest: dict[str, dict[str, object]] = {}
    if workers <= 1:
        for source_path in tqdm(frequency_files, desc="Normalizing frequency files", unit="file"):
            source_name, entry = normalize_single_frequency_file(
                source_path=source_path,
                interpolation_tables=interpolation_tables,
                output_dir=output_dir,
                chunksize=chunksize,
                gzip_compresslevel=gzip_compresslevel,
            )
            manifest[source_name] = entry
        return manifest

    if executor_type == "thread":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    normalize_single_frequency_file,
                    source_path=source_path,
                    interpolation_tables=interpolation_tables,
                    output_dir=output_dir,
                    chunksize=chunksize,
                    gzip_compresslevel=gzip_compresslevel,
                ): source_path
                for source_path in frequency_files
            }
            with tqdm(
                total=len(futures),
                desc="Normalizing frequency files",
                unit="file",
            ) as progress:
                for future in as_completed(futures):
                    source_name, entry = future.result()
                    manifest[source_name] = entry
                    progress.update(1)
                    progress.set_postfix(file=source_name)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_frequency_worker,
            initargs=(interpolation_tables, output_dir, chunksize, gzip_compresslevel),
        ) as executor:
            futures = {
                executor.submit(
                    normalize_single_frequency_file_worker,
                    source_path,
                ): source_path
                for source_path in frequency_files
            }
            with tqdm(
                total=len(futures),
                desc="Normalizing frequency files",
                unit="file",
            ) as progress:
                for future in as_completed(futures):
                    source_name, entry = future.result()
                    manifest[source_name] = entry
                    progress.update(1)
                    progress.set_postfix(file=source_name)

    return manifest


def main() -> None:
    """Run full normalization workflow."""
    # pylint: disable=too-many-locals
    args = parse_args()
    if not 0 <= args.gzip_compresslevel <= 9:
        raise ValueError("--gzip-compresslevel must be between 0 and 9.")
    demand_dir = args.dataset_root / "demand"
    frequency_dir = args.dataset_root / "frequency"
    generation_path = args.dataset_root / "df_fuel_ckan.csv"

    demand_files = sorted(demand_dir.glob("demanddata_*.csv"))
    if not demand_files:
        raise FileNotFoundError(f"No demand files found in {demand_dir}.")

    frequency_files = sorted(
        list(frequency_dir.glob("*.csv")) + list(frequency_dir.glob("*.zip"))
    )
    if not frequency_files:
        raise FileNotFoundError(f"No frequency files found in {frequency_dir}.")

    demand_columns = discover_demand_columns(demand_files)
    demand_table = build_demand_anchor_table(
        demand_files=demand_files,
        demand_columns=demand_columns,
        chunksize=args.demand_chunksize,
    )
    interpolation_tables = build_interpolation_tables(demand_table, prefix="demand_")

    generation_columns = []
    if args.include_generation:
        if not generation_path.exists():
            raise FileNotFoundError(f"Generation mix file not found: {generation_path}")
        generation_table = build_generation_anchor_table(
            path=generation_path,
            chunksize=args.generation_chunksize,
        )
        generation_tables = build_interpolation_tables(generation_table, prefix="generation_")
        interpolation_tables.update(generation_tables)
        generation_columns = list(generation_tables.keys())

    ordered_frequency_files = sorted(frequency_files, key=first_frequency_timestamp)
    manifest = normalize_frequency_files(
        frequency_files=ordered_frequency_files,
        interpolation_tables=interpolation_tables,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
        workers=args.workers,
        executor_type=args.executor,
        gzip_compresslevel=args.gzip_compresslevel,
    )

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "frequency_file_count": len(ordered_frequency_files),
        "demand_feature_count": len(interpolation_tables) - len(generation_columns),
        "generation_feature_count": len(generation_columns),
        "interpolated_feature_count_total": len(interpolation_tables),
        "manifest": manifest,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "normalization_manifest.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("Normalization complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {summary_path}")


if __name__ == "__main__":
    main()
