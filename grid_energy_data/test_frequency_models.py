#!/usr/bin/env python3
"""Evaluate dataset_exploration model families on normalized grid frequency data."""
# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


DEFAULT_MODELS = ["rnn", "tcn", "transformer", "lightgbm", "xgboost"]


@dataclass
class SplitRanges:
    """Exclusive-index row ranges for train, val, and test splits."""

    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    gap_rows: int


@dataclass
class FileCountStats:
    """Per-file row-count and timestamp bounds."""

    path: Path
    rows: int
    min_timestamp_iso: str | None
    max_timestamp_iso: str | None


class FrequencyWindowDataset(Dataset):
    """Sliding-window dataset backed by a disk memmap frequency series."""

    def __init__(
        self,
        values: np.memmap,
        start_index: int,
        end_index: int,
        seq_len: int,
        horizon: int,
        stride: int,
        feature_mean: float,
        feature_std: float,
        target_mean: float,
        target_std: float,
    ) -> None:
        """Initialize a window dataset over one contiguous split.

        :param values: Frequency value memmap.
        :param start_index: Inclusive split start index.
        :param end_index: Exclusive split end index.
        :param seq_len: Input sequence length.
        :param horizon: Prediction horizon in timesteps.
        :param stride: Start-step between adjacent windows.
        :param feature_mean: Train-split feature mean for normalization.
        :param feature_std: Train-split feature std for normalization.
        :param target_mean: Train-split target mean for denormalization.
        :param target_std: Train-split target std for denormalization.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        self.values = values
        self.start_index = start_index
        self.end_index = end_index
        self.seq_len = seq_len
        self.horizon = horizon
        self.stride = stride
        self.feature_mean = np.float32(feature_mean)
        self.feature_std = np.float32(feature_std)
        self.target_mean = np.float32(target_mean)
        self.target_std = np.float32(target_std)

        usable = (end_index - start_index) - seq_len - horizon + 1
        if usable <= 0:
            self.window_count = 0
        else:
            self.window_count = ((usable - 1) // stride) + 1

    def __len__(self) -> int:
        """Return the number of windows in this split.

        :return: Window count.
        """
        return self.window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one normalized feature window and target.

        :param index: Window index.
        :return: Tuple ``(x, y, house_id)`` for compatibility with storenet loops.
        """
        start = self.start_index + index * self.stride
        stop = start + self.seq_len
        target_index = stop + self.horizon - 1

        window = self.values[start:stop]
        normalized_window = (window - self.feature_mean) / self.feature_std
        x = normalized_window.astype(np.float32, copy=False).reshape(-1, 1)

        target = (self.values[target_index] - self.target_mean) / self.target_std
        y = np.array([target], dtype=np.float32)
        house_id = np.array(0, dtype=np.int64)

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(house_id)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Test dataset_exploration model families on normalized frequency data with "
            "leakage-safe train/val/test windows."
        )
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("grid_energy_data/normalized"),
        help="Directory containing normalized partition files with frequency_hz.",
    )
    parser.add_argument(
        "--best-summary",
        type=Path,
        default=Path("dataset_exploration/artifacts/grid_search_summary.json"),
        help="Grid-search summary JSON used to source best params per model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("grid_energy_data/artifacts/frequency_model_test_results.json"),
        help="Destination JSON path for evaluation results.",
    )
    parser.add_argument(
        "--model-artifacts-dir",
        type=Path,
        default=Path("grid_energy_data/artifacts/frequency_model_artifacts"),
        help="Directory where trained model artifacts are saved.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model set to run.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=300_000,
        help="Chunk size used while reading normalized partitions.",
    )
    parser.add_argument(
        "--count-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel worker count for normalized row counting.",
    )
    parser.add_argument(
        "--materialize-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel worker count for memmap materialization.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5_000_000,
        help=(
            "Maximum recent rows kept from normalized data. Set <=0 to use all rows; "
            "capping rows controls memory/runtime."
        ),
    )
    parser.add_argument(
        "--max-neural-windows-per-split",
        type=int,
        default=300_000,
        help="Cap for windows used per split in neural training/evaluation.",
    )
    parser.add_argument(
        "--max-tabular-windows-per-split",
        type=int,
        default=200_000,
        help="Cap for windows used per split in tree-based model training/evaluation.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=None,
        help="Override train fraction from best-summary data params.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Override validation fraction from best-summary data params.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Dataloader worker count for neural models.",
    )
    return parser.parse_args()


def parse_model_selection(raw_models: str) -> list[str]:
    """Parse a comma-separated model list.

    :param raw_models: Raw `--models` argument.
    :return: Validated model list.
    """
    models = [name.strip().lower() for name in raw_models.split(",") if name.strip()]
    unsupported = [name for name in models if name not in DEFAULT_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported model(s): {unsupported}")
    return models


def first_normalized_timestamp(path: Path) -> pd.Timestamp:
    """Read the first timestamp in one normalized partition.

    :param path: Normalized partition path.
    :return: First parseable UTC timestamp.
    """
    sample = pd.read_csv(
        path,
        compression="infer",
        usecols=["timestamp_utc"],
        nrows=32,
        dtype="string",
    )
    parsed = pd.to_datetime(sample["timestamp_utc"], errors="coerce", utc=True)
    valid = parsed.dropna()
    if valid.empty:
        raise ValueError(f"No valid timestamp found in file head: {path}")
    return valid.iloc[0]


def list_normalized_files(normalized_dir: Path) -> list[Path]:
    """Locate and timestamp-sort normalized partition files.

    :param normalized_dir: Directory containing normalized CSV partitions.
    :return: Sorted normalized file list.
    """
    files = sorted(normalized_dir.glob("*_normalized.csv.gz"))
    if not files:
        files = sorted(normalized_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No normalized .csv.gz files found in {normalized_dir}")
    return sorted(files, key=first_normalized_timestamp)


def scan_normalized_row_count(files: list[Path], chunksize: int) -> tuple[int, str, str]:
    """Scan normalized files and count valid frequency rows.

    :param files: Sorted normalized partition files.
    :param chunksize: Chunk size for streamed CSV reads.
    :return: Tuple of `(valid_row_count, min_timestamp_iso, max_timestamp_iso)`.
    """
    return scan_normalized_row_count_parallel(files, chunksize, workers=1)


def count_single_normalized_file(
    path: Path,
    chunksize: int,
) -> tuple[str, int, str | None, str | None]:
    """Count valid rows and timestamp range for one normalized file.

    :param path: Normalized CSV file path.
    :param chunksize: Chunk size for streamed reads.
    :return: Tuple ``(filename, rows, min_timestamp_iso, max_timestamp_iso)``.
    """
    rows = 0
    min_timestamp = None
    max_timestamp = None

    reader = pd.read_csv(
        path,
        compression="infer",
        usecols=["timestamp_utc", "frequency_hz"],
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        timestamps = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True)
        frequency = pd.to_numeric(chunk["frequency_hz"], errors="coerce")
        valid = timestamps.notna() & frequency.notna()
        if not valid.any():
            continue

        valid_timestamps = timestamps.loc[valid]
        rows += int(valid.sum())
        chunk_min = valid_timestamps.iloc[0]
        chunk_max = valid_timestamps.iloc[-1]
        min_timestamp = chunk_min if min_timestamp is None else min(min_timestamp, chunk_min)
        max_timestamp = chunk_max if max_timestamp is None else max(max_timestamp, chunk_max)

    return (
        path.name,
        rows,
        None if min_timestamp is None else min_timestamp.isoformat(),
        None if max_timestamp is None else max_timestamp.isoformat(),
    )


def count_files_parallel(
    files: list[Path],
    chunksize: int,
    workers: int,
) -> list[FileCountStats]:
    """Count valid frequency rows in each normalized file.

    :param files: Sorted normalized partition files.
    :param chunksize: Chunk size for streamed CSV reads.
    :param workers: Number of process workers.
    :return: Per-file count stats in the same order as `files`.
    """
    by_name: dict[str, FileCountStats] = {}

    if workers <= 1:
        for path in tqdm(files, desc="Counting normalized rows", unit="file"):
            file_name, rows, min_iso, max_iso = count_single_normalized_file(path, chunksize)
            by_name[file_name] = FileCountStats(
                path=path,
                rows=rows,
                min_timestamp_iso=min_iso,
                max_timestamp_iso=max_iso,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(count_single_normalized_file, path, chunksize): path
                for path in files
            }
            with tqdm(total=len(futures), desc="Counting normalized rows", unit="file") as progress:
                for future in as_completed(futures):
                    path = futures[future]
                    file_name, rows, min_iso, max_iso = future.result()
                    by_name[file_name] = FileCountStats(
                        path=path,
                        rows=rows,
                        min_timestamp_iso=min_iso,
                        max_timestamp_iso=max_iso,
                    )
                    progress.update(1)
                    progress.set_postfix(file=file_name)

    return [by_name[path.name] for path in files]


def scan_normalized_row_count_parallel(
    files: list[Path],
    chunksize: int,
    workers: int,
) -> tuple[int, str, str]:
    """Scan normalized files and count valid frequency rows in parallel.

    :param files: Sorted normalized partition files.
    :param chunksize: Chunk size for streamed CSV reads.
    :param workers: Number of process workers.
    :return: Tuple of `(valid_row_count, min_timestamp_iso, max_timestamp_iso)`.
    """
    # pylint: disable=too-many-locals
    total_rows = 0
    min_timestamp = None
    max_timestamp = None

    stats = count_files_parallel(files=files, chunksize=chunksize, workers=workers)
    for file_stats in stats:
        total_rows += file_stats.rows
        if file_stats.min_timestamp_iso is not None:
            min_ts = pd.Timestamp(file_stats.min_timestamp_iso)
            max_ts = pd.Timestamp(file_stats.max_timestamp_iso)
            min_timestamp = min_ts if min_timestamp is None else min(min_timestamp, min_ts)
            max_timestamp = max_ts if max_timestamp is None else max(max_timestamp, max_ts)

    if total_rows == 0 or min_timestamp is None or max_timestamp is None:
        raise RuntimeError("No valid frequency rows found in normalized files.")

    return total_rows, min_timestamp.isoformat(), max_timestamp.isoformat()


def extract_frequency_slice(
    path: Path,
    chunksize: int,
    skip_rows: int,
    take_rows: int,
) -> np.ndarray:
    """Extract exactly `take_rows` valid frequency values after skipping `skip_rows`.

    :param path: Source normalized file.
    :param chunksize: Chunk size for streamed reads.
    :param skip_rows: Number of valid rows to skip from file start.
    :param take_rows: Number of valid rows to extract.
    :return: Extracted float32 frequency vector.
    """
    collected: list[np.ndarray] = []
    skipped = 0
    taken = 0
    reader = pd.read_csv(
        path,
        compression="infer",
        usecols=["timestamp_utc", "frequency_hz"],
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        timestamps = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True)
        frequency = pd.to_numeric(chunk["frequency_hz"], errors="coerce")
        valid = timestamps.notna() & frequency.notna()
        if not valid.any():
            continue

        valid_frequency = frequency.loc[valid].to_numpy(dtype=np.float32)
        if skipped < skip_rows:
            delta = min(skip_rows - skipped, len(valid_frequency))
            valid_frequency = valid_frequency[delta:]
            skipped += delta
            if len(valid_frequency) == 0:
                continue

        remaining = take_rows - taken
        if remaining <= 0:
            break
        if len(valid_frequency) > remaining:
            valid_frequency = valid_frequency[:remaining]
        collected.append(valid_frequency)
        taken += len(valid_frequency)
        if taken >= take_rows:
            break

    if taken != take_rows:
        raise RuntimeError(
            f"Failed to extract expected rows from {path}: got {taken}, need {take_rows}"
        )
    if not collected:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def materialize_single_file_to_memmap(
    task: tuple[Path, int, int, int],
    memmap_path: Path,
    kept_rows: int,
    chunksize: int,
) -> tuple[str, int]:
    """Execute one file write task into the shared memmap path.

    :param task: Tuple `(path, skip_rows, take_rows, dest_offset)`.
    :param memmap_path: Target memmap path.
    :param kept_rows: Full memmap row count.
    :param chunksize: Chunk size for streamed reads.
    :return: Tuple `(source_filename, rows_written)`.
    """
    path, skip_rows, take_rows, dest_offset = task
    slice_values = extract_frequency_slice(
        path=path,
        chunksize=chunksize,
        skip_rows=skip_rows,
        take_rows=take_rows,
    )
    mmap = np.memmap(memmap_path, dtype=np.float32, mode="r+", shape=(kept_rows,))
    mmap[dest_offset : dest_offset + take_rows] = slice_values
    mmap.flush()
    return path.name, take_rows


def materialize_frequency_memmap(
    files: list[Path],
    chunksize: int,
    count_workers: int,
    materialize_workers: int,
    max_rows: int,
    output_dir: Path,
) -> tuple[Path, int]:
    """Stream normalized frequency values into a disk memmap.

    :param files: Sorted normalized partition files.
    :param chunksize: Chunk size for streamed CSV reads.
    :param count_workers: Worker count used during row counting.
    :param materialize_workers: Worker count used during memmap writing.
    :param max_rows: Max rows to keep; keep the most recent rows when limited.
    :param output_dir: Directory where memmap file is created.
    :return: Tuple `(memmap_path, row_count_kept)`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    file_stats = count_files_parallel(files=files, chunksize=chunksize, workers=count_workers)
    total_rows = int(sum(stats.rows for stats in file_stats))
    if total_rows == 0:
        raise RuntimeError("No valid frequency rows found in normalized files.")
    if max_rows > 0:
        kept_rows = min(total_rows, max_rows)
    else:
        kept_rows = total_rows
    rows_to_skip = total_rows - kept_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = output_dir / "frequency_values.float32.mmap"
    if memmap_path.exists():
        memmap_path.unlink()
    values = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(kept_rows,))
    values.flush()

    def plan_file_writes() -> list[tuple[Path, int, int, int]]:
        """Build per-file write tasks as `(path, skip_rows, take_rows, dest_offset)`."""
        tasks: list[tuple[Path, int, int, int]] = []
        full_start = 0
        keep_start = rows_to_skip
        keep_end = rows_to_skip + kept_rows
        for stats in file_stats:
            full_end = full_start + stats.rows
            overlap_start = max(full_start, keep_start)
            overlap_end = min(full_end, keep_end)
            if overlap_end > overlap_start:
                skip_rows = overlap_start - full_start
                take_rows = overlap_end - overlap_start
                dest_offset = overlap_start - keep_start
                tasks.append((stats.path, skip_rows, take_rows, dest_offset))
            full_start = full_end
        return tasks

    tasks = plan_file_writes()
    written_total = 0
    if materialize_workers <= 1:
        for task in tqdm(tasks, desc="Materializing frequency series", unit="file"):
            _, row_count = materialize_single_file_to_memmap(
                task=task,
                memmap_path=memmap_path,
                kept_rows=kept_rows,
                chunksize=chunksize,
            )
            written_total += row_count
    else:
        with ProcessPoolExecutor(max_workers=materialize_workers) as executor:
            futures = {
                executor.submit(
                    materialize_single_file_to_memmap,
                    task,
                    memmap_path,
                    kept_rows,
                    chunksize,
                ): task
                for task in tasks
            }
            with tqdm(
                total=len(futures),
                desc="Materializing frequency series",
                unit="file",
            ) as progress:
                for future in as_completed(futures):
                    file_name, row_count = future.result()
                    written_total += row_count
                    progress.update(1)
                    progress.set_postfix(file=file_name)

    if written_total != kept_rows:
        raise RuntimeError(
            f"Frequency memmap write incomplete: wrote {written_total}, expected {kept_rows}"
        )
    return memmap_path, kept_rows


def load_best_summary(path: Path) -> dict[str, Any]:
    """Load the best-parameter summary JSON.

    :param path: Summary JSON path.
    :return: Parsed summary dictionary.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_params(
    summary: dict[str, Any],
    train_frac: float | None,
    val_frac: float | None,
) -> dict:
    """Resolve data parameters used for all models.

    :param summary: Grid-search summary JSON payload.
    :param train_frac: Optional train fraction override.
    :param val_frac: Optional validation fraction override.
    :return: Data-parameter dictionary.
    """
    best_by_model = summary.get("best_by_model", {})
    if not best_by_model:
        raise ValueError("best_by_model is missing or empty in the summary.")

    first_model = next(iter(best_by_model.values()))
    data_params = dict(first_model["data_params"])
    if train_frac is not None:
        data_params["train_frac"] = train_frac
    if val_frac is not None:
        data_params["val_frac"] = val_frac
    return data_params


def compute_split_ranges(
    total_rows: int,
    train_frac: float,
    val_frac: float,
    seq_len: int,
    horizon: int,
) -> SplitRanges:
    """Create split ranges with explicit guard gaps for leakage prevention.

    :param total_rows: Number of rows in the working frequency series.
    :param train_frac: Fraction allocated to train rows.
    :param val_frac: Fraction allocated to validation rows.
    :param seq_len: Input sequence length.
    :param horizon: Prediction horizon.
    :return: `SplitRanges` with gap-separated ranges.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0")

    train_end = int(total_rows * train_frac)
    val_end = int(total_rows * (train_frac + val_frac))
    gap_rows = seq_len + horizon - 1

    val_start = train_end + gap_rows
    test_start = val_end + gap_rows

    ranges = SplitRanges(
        train_start=0,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=total_rows,
        gap_rows=gap_rows,
    )

    if ranges.val_start >= ranges.val_end:
        raise ValueError("Validation split became empty after applying leakage guard gap.")
    if ranges.test_start >= ranges.test_end:
        raise ValueError("Test split became empty after applying leakage guard gap.")
    return ranges


def split_window_count(
    start_index: int,
    end_index: int,
    seq_len: int,
    horizon: int,
    stride: int,
) -> int:
    """Compute how many sliding windows exist in one split.

    :param start_index: Inclusive split start.
    :param end_index: Exclusive split end.
    :param seq_len: Window length.
    :param horizon: Prediction horizon.
    :param stride: Window stride.
    :return: Number of valid windows.
    """
    usable = (end_index - start_index) - seq_len - horizon + 1
    if usable <= 0:
        return 0
    return ((usable - 1) // stride) + 1


def evenly_spaced_indices(total: int, max_count: int) -> np.ndarray | None:
    """Return an evenly spaced index subset when capping sample count.

    :param total: Total available count.
    :param max_count: Maximum desired count.
    :return: Index array or `None` when no capping is needed.
    """
    if max_count <= 0 or total <= max_count:
        return None
    return np.linspace(0, total - 1, num=max_count, dtype=np.int64)


def maybe_subset_dataset(dataset: Dataset, max_windows: int) -> Dataset:
    """Apply even subsampling to a dataset when requested.

    :param dataset: Base dataset.
    :param max_windows: Maximum windows allowed.
    :return: Original or subsetted dataset.
    """
    subset_indices = evenly_spaced_indices(len(dataset), max_windows)
    if subset_indices is None:
        return dataset
    return Subset(dataset, subset_indices.tolist())


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a dataloader with sensible defaults.

    :param dataset: Dataset to wrap.
    :param batch_size: Batch size.
    :param shuffle: Shuffle setting.
    :param num_workers: Worker process count.
    :return: Configured dataloader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 for one target column.

    :param predictions: Predicted values, shape `(n, 1)` or `(n,)`.
    :param targets: Ground-truth values, shape `(n, 1)` or `(n,)`.
    :return: Regression metric dictionary.
    """
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    true = np.asarray(targets, dtype=np.float64).reshape(-1)

    errors = pred - true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    ss_res = float(np.sum(np.square(errors)))
    centered = true - np.mean(true)
    ss_tot = float(np.sum(np.square(centered)))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - (ss_res / ss_tot))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def build_tabular_windows(
    values: np.memmap,
    start_index: int,
    end_index: int,
    seq_len: int,
    horizon: int,
    stride: int,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flattened lag windows for tree-based models.

    :param values: Frequency series values.
    :param start_index: Inclusive split start.
    :param end_index: Exclusive split end.
    :param seq_len: Window length.
    :param horizon: Prediction horizon.
    :param stride: Window stride.
    :param max_windows: Window cap for this split.
    :return: Tuple `(x, y)` where `x` is shape `(n, seq_len)` and `y` shape `(n,)`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    total_windows = split_window_count(start_index, end_index, seq_len, horizon, stride)
    if total_windows == 0:
        return (
            np.empty((0, seq_len), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    pick = evenly_spaced_indices(total_windows, max_windows)
    if pick is None:
        pick = np.arange(total_windows, dtype=np.int64)

    x = np.empty((len(pick), seq_len), dtype=np.float32)
    y = np.empty((len(pick),), dtype=np.float32)

    for out_index, window_offset in enumerate(pick):
        start = start_index + int(window_offset) * stride
        stop = start + seq_len
        target_index = stop + horizon - 1
        x[out_index, :] = values[start:stop]
        y[out_index] = values[target_index]

    return x, y


def import_storenet_helpers() -> dict[str, Any]:
    """Import reusable training/model helpers from dataset_exploration.

    :return: Mapping containing imported helper objects.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # pylint: disable=import-outside-toplevel,import-error
    from dataset_exploration.storenet_ml import config as st_config

    st_config.TARGET_COLUMNS[:] = ["frequency_hz"]

    from dataset_exploration.storenet_ml.models import (
        SharedEnergyRNN,
        SharedEnergyTCN,
        SharedEnergyTransformer,
    )
    from dataset_exploration.storenet_ml.training import (
        collect_predictions,
        set_seed,
        train_one_epoch,
    )

    return {
        "SharedEnergyRNN": SharedEnergyRNN,
        "SharedEnergyTCN": SharedEnergyTCN,
        "SharedEnergyTransformer": SharedEnergyTransformer,
        "collect_predictions": collect_predictions,
        "set_seed": set_seed,
        "train_one_epoch": train_one_epoch,
    }


def train_and_eval_neural(
    model_name: str,
    model: nn.Module,
    model_params: dict[str, Any],
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    collect_predictions,
    train_one_epoch,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    num_workers: int,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate one neural model with early stopping.

    :param model_name: Model label.
    :param model: Initialized model instance.
    :param model_params: Best hyperparameters for this model.
    :param train_dataset: Training dataset.
    :param val_dataset: Validation dataset.
    :param test_dataset: Test dataset.
    :param collect_predictions: Imported prediction helper.
    :param train_one_epoch: Imported one-epoch training helper.
    :param target_mean: Denormalization mean array.
    :param target_std: Denormalization std array.
    :param device: Computation device.
    :param num_workers: Dataloader worker count.
    :param artifact_path: Optional destination for serialized best checkpoint.
    :return: Metrics payload for this model.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(f"{model_name}: one or more splits have zero windows.")

    train_loader = build_loader(
        train_dataset,
        batch_size=int(model_params["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=int(model_params["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=int(model_params["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_params["learning_rate"]),
        weight_decay=float(model_params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_val_metrics = None
    patience_left = int(model_params["patience"])

    epochs = int(model_params["epochs"])
    trained_epochs = 0
    epoch_progress = tqdm(range(1, epochs + 1), desc=f"{model_name} training", leave=False)
    for epoch in epoch_progress:
        trained_epochs = epoch
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        val_loss, val_predictions, val_targets = collect_predictions(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            desc=f"{model_name} epoch {epoch} [val]",
        )
        val_metrics = regression_metrics(val_predictions, val_targets)
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_mae=f"{val_metrics['mae']:.5f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = int(model_params["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError(f"{model_name}: no valid checkpoint was produced.")

    saved_artifact = None
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": model_name,
                "model_params": model_params,
                "best_state_dict": best_state,
                "feature_mean": float(target_mean[0]),
                "feature_std": float(target_std[0]),
            },
            artifact_path,
        )
        saved_artifact = str(artifact_path.resolve())

    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_predictions, test_targets = collect_predictions(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        desc=f"{model_name} [test]",
    )
    test_metrics = regression_metrics(test_predictions, test_targets)
    test_metrics["test_loss"] = float(test_loss)

    return {
        "status": "ok",
        "score": float(best_val_metrics["mae"]),
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "trained_epochs": int(trained_epochs),
        "artifact_path": saved_artifact,
    }


def main() -> None:
    """Run leakage-safe frequency prediction tests for selected model families."""
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    args = parse_args()
    selected_models = parse_model_selection(args.models)
    best_summary = load_best_summary(args.best_summary)
    data_params = resolve_data_params(
        best_summary,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    seq_len = int(data_params["seq_len"])
    horizon = int(data_params["horizon"])
    stride = int(data_params["stride"])
    train_frac = float(data_params["train_frac"])
    val_frac = float(data_params["val_frac"])

    normalized_files = list_normalized_files(args.normalized_dir)
    artifacts_dir = args.output.parent
    memmap_path, kept_rows = materialize_frequency_memmap(
        files=normalized_files,
        chunksize=args.chunksize,
        count_workers=args.count_workers,
        materialize_workers=args.materialize_workers,
        max_rows=args.max_rows,
        output_dir=artifacts_dir,
    )
    values = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(kept_rows,))

    ranges = compute_split_ranges(
        total_rows=kept_rows,
        train_frac=train_frac,
        val_frac=val_frac,
        seq_len=seq_len,
        horizon=horizon,
    )

    train_window_count = split_window_count(
        ranges.train_start,
        ranges.train_end,
        seq_len,
        horizon,
        stride,
    )
    val_window_count = split_window_count(
        ranges.val_start,
        ranges.val_end,
        seq_len,
        horizon,
        stride,
    )
    test_window_count = split_window_count(
        ranges.test_start,
        ranges.test_end,
        seq_len,
        horizon,
        stride,
    )
    if min(train_window_count, val_window_count, test_window_count) <= 0:
        raise RuntimeError("At least one split has zero valid windows.")

    helper = import_storenet_helpers()
    helper["set_seed"](args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_values = np.asarray(values[ranges.train_start : ranges.train_end], dtype=np.float32)
    feature_mean = float(np.mean(train_values))
    feature_std = float(np.std(train_values))
    if feature_std < 1e-6:
        feature_std = 1.0
    target_mean = np.array([feature_mean], dtype=np.float32)
    target_std = np.array([feature_std], dtype=np.float32)

    train_dataset = FrequencyWindowDataset(
        values=values,
        start_index=ranges.train_start,
        end_index=ranges.train_end,
        seq_len=seq_len,
        horizon=horizon,
        stride=stride,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=feature_mean,
        target_std=feature_std,
    )
    val_dataset = FrequencyWindowDataset(
        values=values,
        start_index=ranges.val_start,
        end_index=ranges.val_end,
        seq_len=seq_len,
        horizon=horizon,
        stride=stride,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=feature_mean,
        target_std=feature_std,
    )
    test_dataset = FrequencyWindowDataset(
        values=values,
        start_index=ranges.test_start,
        end_index=ranges.test_end,
        seq_len=seq_len,
        horizon=horizon,
        stride=stride,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=feature_mean,
        target_std=feature_std,
    )

    neural_train = maybe_subset_dataset(train_dataset, args.max_neural_windows_per_split)
    neural_val = maybe_subset_dataset(val_dataset, args.max_neural_windows_per_split)
    neural_test = maybe_subset_dataset(test_dataset, args.max_neural_windows_per_split)

    best_by_model = best_summary.get("best_by_model", {})
    results: dict[str, Any] = {}

    for model_name in selected_models:
        if model_name not in best_by_model:
            results[model_name] = {
                "status": "error",
                "error_type": "MissingBestParams",
                "error": f"best_by_model['{model_name}'] not found in summary.",
            }
            continue

        model_params = dict(best_by_model[model_name]["model_params"])

        try:
            if model_name == "rnn":
                model = helper["SharedEnergyRNN"](
                    feature_dim=1,
                    num_houses=1,
                    hidden_size=int(model_params["hidden_size"]),
                    house_embedding_dim=int(model_params["house_embedding_dim"]),
                    num_layers=int(model_params["num_layers"]),
                    dropout=float(model_params["dropout"]),
                ).to(device)
                results[model_name] = train_and_eval_neural(
                    model_name=model_name,
                    model=model,
                    model_params=model_params,
                    train_dataset=neural_train,
                    val_dataset=neural_val,
                    test_dataset=neural_test,
                    collect_predictions=helper["collect_predictions"],
                    train_one_epoch=helper["train_one_epoch"],
                    target_mean=target_mean,
                    target_std=target_std,
                    device=device,
                    num_workers=args.num_workers,
                    artifact_path=args.model_artifacts_dir / f"{model_name}_best.pt",
                )
            elif model_name == "tcn":
                model = helper["SharedEnergyTCN"](
                    feature_dim=1,
                    num_houses=1,
                    hidden_size=int(model_params["hidden_size"]),
                    house_embedding_dim=int(model_params["house_embedding_dim"]),
                    num_layers=int(model_params["num_layers"]),
                    kernel_size=int(model_params["kernel_size"]),
                    dropout=float(model_params["dropout"]),
                ).to(device)
                results[model_name] = train_and_eval_neural(
                    model_name=model_name,
                    model=model,
                    model_params=model_params,
                    train_dataset=neural_train,
                    val_dataset=neural_val,
                    test_dataset=neural_test,
                    collect_predictions=helper["collect_predictions"],
                    train_one_epoch=helper["train_one_epoch"],
                    target_mean=target_mean,
                    target_std=target_std,
                    device=device,
                    num_workers=args.num_workers,
                    artifact_path=args.model_artifacts_dir / f"{model_name}_best.pt",
                )
            elif model_name == "transformer":
                model = helper["SharedEnergyTransformer"](
                    feature_dim=1,
                    num_houses=1,
                    max_seq_len=seq_len,
                    d_model=int(model_params["d_model"]),
                    house_embedding_dim=int(model_params["house_embedding_dim"]),
                    num_layers=int(model_params["num_layers"]),
                    num_heads=int(model_params["num_heads"]),
                    feedforward_dim=int(model_params["feedforward_dim"]),
                    dropout=float(model_params["dropout"]),
                ).to(device)
                results[model_name] = train_and_eval_neural(
                    model_name=model_name,
                    model=model,
                    model_params=model_params,
                    train_dataset=neural_train,
                    val_dataset=neural_val,
                    test_dataset=neural_test,
                    collect_predictions=helper["collect_predictions"],
                    train_one_epoch=helper["train_one_epoch"],
                    target_mean=target_mean,
                    target_std=target_std,
                    device=device,
                    num_workers=args.num_workers,
                    artifact_path=args.model_artifacts_dir / f"{model_name}_best.pt",
                )
            elif model_name == "lightgbm":
                if LGBMRegressor is None:
                    raise ImportError("lightgbm is not installed.")
                x_train, y_train = build_tabular_windows(
                    values=values,
                    start_index=ranges.train_start,
                    end_index=ranges.train_end,
                    seq_len=seq_len,
                    horizon=horizon,
                    stride=stride,
                    max_windows=args.max_tabular_windows_per_split,
                )
                x_val, y_val = build_tabular_windows(
                    values=values,
                    start_index=ranges.val_start,
                    end_index=ranges.val_end,
                    seq_len=seq_len,
                    horizon=horizon,
                    stride=stride,
                    max_windows=args.max_tabular_windows_per_split,
                )
                x_test, y_test = build_tabular_windows(
                    values=values,
                    start_index=ranges.test_start,
                    end_index=ranges.test_end,
                    seq_len=seq_len,
                    horizon=horizon,
                    stride=stride,
                    max_windows=args.max_tabular_windows_per_split,
                )
                model = LGBMRegressor(
                    objective="regression",
                    n_estimators=int(model_params["n_estimators"]),
                    learning_rate=float(model_params["learning_rate"]),
                    num_leaves=int(model_params["num_leaves"]),
                    max_depth=int(model_params["max_depth"]),
                    subsample=float(model_params["subsample"]),
                    colsample_bytree=float(model_params["colsample_bytree"]),
                    min_child_samples=int(model_params["min_child_samples"]),
                    reg_alpha=float(model_params["reg_alpha"]),
                    reg_lambda=float(model_params["reg_lambda"]),
                    device_type=str(model_params["device_type"]),
                    random_state=args.seed,
                    n_jobs=-1,
                    verbosity=-1,
                )
                model.fit(x_train, y_train)
                artifact_path = args.model_artifacts_dir / f"{model_name}_best.pkl"
                with artifact_path.open("wb") as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                val_pred = model.predict(x_val)
                test_pred = model.predict(x_test)
                val_metrics = regression_metrics(val_pred, y_val)
                test_metrics = regression_metrics(test_pred, y_test)
                results[model_name] = {
                    "status": "ok",
                    "score": float(val_metrics["mae"]),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "artifact_path": str(artifact_path.resolve()),
                }
            elif model_name == "xgboost":
                if XGBRegressor is None:
                    raise ImportError("xgboost is not installed.")
                x_train, y_train = build_tabular_windows(
                    values=values,
                    start_index=ranges.train_start,
                    end_index=ranges.train_end,
                    seq_len=seq_len,
                    horizon=horizon,
                    stride=stride,
                    max_windows=args.max_tabular_windows_per_split,
                )
                x_val, y_val = build_tabular_windows(
                    values=values,
                    start_index=ranges.val_start,
                    end_index=ranges.val_end,
                    seq_len=seq_len,
                    horizon=horizon,
                    stride=stride,
                    max_windows=args.max_tabular_windows_per_split,
                )
                x_test, y_test = build_tabular_windows(
                    values=values,
                    start_index=ranges.test_start,
                    end_index=ranges.test_end,
                    seq_len=seq_len,
                    horizon=horizon,
                    stride=stride,
                    max_windows=args.max_tabular_windows_per_split,
                )
                xgb_params = dict(model_params)
                if (
                    str(xgb_params.get("device", "cpu")).startswith("cuda")
                    and not torch.cuda.is_available()
                ):
                    xgb_params["device"] = "cpu"
                model = XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=int(xgb_params["n_estimators"]),
                    learning_rate=float(xgb_params["learning_rate"]),
                    max_depth=int(xgb_params["max_depth"]),
                    min_child_weight=float(xgb_params["min_child_weight"]),
                    subsample=float(xgb_params["subsample"]),
                    colsample_bytree=float(xgb_params["colsample_bytree"]),
                    reg_alpha=float(xgb_params["reg_alpha"]),
                    reg_lambda=float(xgb_params["reg_lambda"]),
                    tree_method=str(xgb_params["tree_method"]),
                    device=str(xgb_params["device"]),
                    random_state=args.seed,
                    n_jobs=-1,
                )
                model.fit(x_train, y_train)
                artifact_path = args.model_artifacts_dir / f"{model_name}_best.pkl"
                with artifact_path.open("wb") as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                val_pred = model.predict(x_val)
                test_pred = model.predict(x_test)
                val_metrics = regression_metrics(val_pred, y_val)
                test_metrics = regression_metrics(test_pred, y_test)
                results[model_name] = {
                    "status": "ok",
                    "score": float(val_metrics["mae"]),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "resolved_device": str(xgb_params["device"]),
                    "artifact_path": str(artifact_path.resolve()),
                }
            else:
                raise ValueError(f"Unhandled model name: {model_name}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            results[model_name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    payload = {
        "data_params": data_params,
        "normalization_input_dir": str(args.normalized_dir.resolve()),
        "best_summary": str(args.best_summary.resolve()),
        "selected_models": selected_models,
        "device": str(device),
        "series_rows_total_used": kept_rows,
        "split_ranges": {
            "train": [ranges.train_start, ranges.train_end],
            "val": [ranges.val_start, ranges.val_end],
            "test": [ranges.test_start, ranges.test_end],
            "gap_rows": ranges.gap_rows,
        },
        "window_counts_before_caps": {
            "train": train_window_count,
            "val": val_window_count,
            "test": test_window_count,
        },
        "window_caps": {
            "max_neural_windows_per_split": args.max_neural_windows_per_split,
            "max_tabular_windows_per_split": args.max_tabular_windows_per_split,
        },
        "leakage_guard": (
            "Raw split ranges are separated by a gap of seq_len+horizon-1 rows so train/val/test "
            "windows cannot overlap in input rows or target rows."
        ),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
