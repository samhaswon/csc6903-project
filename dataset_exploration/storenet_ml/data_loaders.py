from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from storenet_ml.config import DATA_DIR, ENERGY_COLUMNS, INPUT_FEATURES, TARGET_COLUMNS, WEATHER_COLUMNS


@dataclass
class StandardizationStats:
    """Container for dataset scaling statistics."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def strip_and_coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Strip whitespace and coerce selected columns to numeric values.

    :param frame: Input dataframe to update in place.
    :param columns: Column names to coerce.
    :return: Dataframe with numeric-converted columns.
    """
    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column].astype("string").str.strip(),
            errors="coerce",
        )
    return frame


def interpolate_short_gaps(
    frame: pd.DataFrame,
    columns: list[str],
    max_interp_gap: int,
) -> pd.DataFrame:
    """Interpolate short missing spans for selected columns.

    :param frame: Dataframe with missing values.
    :param columns: Columns that should be interpolated.
    :param max_interp_gap: Maximum gap length to fill.
    :return: Dataframe with interpolated values.
    """
    frame[columns] = frame[columns].interpolate(
        method="linear",
        limit=max_interp_gap,
        limit_direction="both",
    )
    return frame


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar features derived from the ``date`` column.

    :param frame: Dataframe containing a datetime ``date`` column.
    :return: Dataframe with sine and cosine calendar features.
    """
    minutes_of_day = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    day_of_week = frame["date"].dt.dayofweek
    day_of_year = frame["date"].dt.dayofyear

    frame["hour_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
    frame["hour_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440.0)
    frame["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    frame["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    frame["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    return frame


def load_weather(max_interp_gap: int) -> pd.DataFrame:
    """Load, clean, and interpolate weather data.

    :param max_interp_gap: Maximum number of missing minutes to interpolate.
    :return: Clean weather dataframe indexed by minute-level timestamps.
    """
    weather = pd.read_csv(
        DATA_DIR / "weather.csv",
        dtype="string",
        low_memory=False,
        na_values=["", " ", "  ", "   "],
    )
    weather.columns = weather.columns.str.strip()
    weather["date"] = pd.to_datetime(
        weather["date"].astype("string").str.strip(),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    weather = strip_and_coerce_numeric(weather, WEATHER_COLUMNS)
    weather = weather.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    weather = weather.set_index("date")
    full_index = pd.date_range(weather.index.min(), weather.index.max(), freq="1min")
    weather = weather.reindex(full_index)
    weather.index.name = "date"
    weather = interpolate_short_gaps(weather, WEATHER_COLUMNS, max_interp_gap)
    weather = weather.dropna(subset=WEATHER_COLUMNS)
    return weather


def load_house_frame(energy_path, weather: pd.DataFrame, max_interp_gap: int) -> pd.DataFrame:
    """Load one house energy file and join it with weather features.

    :param energy_path: Path to a house energy CSV file.
    :param weather: Preprocessed weather dataframe.
    :param max_interp_gap: Maximum number of missing minutes to interpolate.
    :return: Clean merged frame with engineered calendar and house features.
    """
    frame = pd.read_csv(
        energy_path,
        dtype="string",
        low_memory=False,
        na_values=["", " ", "  ", "   "],
    )
    frame.columns = frame.columns.str.strip()
    frame["date"] = pd.to_datetime(
        frame["date"].astype("string").str.strip(),
        errors="coerce",
    )
    frame = strip_and_coerce_numeric(frame, ENERGY_COLUMNS)
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    frame = frame.set_index("date")

    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="1min")
    frame = frame.reindex(full_index)
    frame.index.name = "date"
    frame = interpolate_short_gaps(frame, ENERGY_COLUMNS, max_interp_gap)

    merged = frame.join(weather, how="left")
    merged = merged.dropna(subset=ENERGY_COLUMNS + WEATHER_COLUMNS).reset_index()
    merged = merged.rename(columns={"index": "date"})
    merged = add_calendar_features(merged)
    merged["house_id"] = int(energy_path.stem.split("_")[0][1:]) - 1
    merged["house_name"] = energy_path.stem.split("_")[0]
    return merged


def split_house_frame(
    frame: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> dict[str, pd.DataFrame]:
    """Split a house frame into contiguous train/validation/test portions.

    :param frame: Full house dataframe sorted by time.
    :param train_frac: Fraction of rows assigned to the train split.
    :param val_frac: Fraction of rows assigned to the validation split.
    :return: Mapping with ``train``, ``val``, and ``test`` dataframes.
    """
    n_rows = len(frame)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return {
        "train": frame.iloc[:train_end].reset_index(drop=True),
        "val": frame.iloc[train_end:val_end].reset_index(drop=True),
        "test": frame.iloc[val_end:].reset_index(drop=True),
    }


def update_running_stats(
    frame: pd.DataFrame,
    columns: list[str],
    total_rows: int,
    total_sum: np.ndarray,
    total_sum_sq: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Update running row-count, sum, and squared-sum statistics.

    :param frame: Dataframe chunk to aggregate.
    :param columns: Numeric columns to include.
    :param total_rows: Current cumulative row count.
    :param total_sum: Current cumulative column sums.
    :param total_sum_sq: Current cumulative squared column sums.
    :return: Updated ``(total_rows, total_sum, total_sum_sq)`` tuple.
    """
    values = frame[columns].to_numpy(dtype=np.float64)
    total_rows += values.shape[0]
    total_sum += values.sum(axis=0)
    total_sum_sq += np.square(values).sum(axis=0)
    return total_rows, total_sum, total_sum_sq


def finalize_running_stats(
    total_rows: int,
    total_sum: np.ndarray,
    total_sum_sq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert running totals into mean and standard deviation arrays.

    :param total_rows: Number of aggregated rows.
    :param total_sum: Cumulative column sums.
    :param total_sum_sq: Cumulative column squared sums.
    :return: Tuple of ``(mean, std)`` float32 arrays.
    """
    mean = total_sum / total_rows
    variance = (total_sum_sq / total_rows) - np.square(mean)
    std = np.sqrt(np.clip(variance, 1e-8, None))
    return mean.astype(np.float32), std.astype(np.float32)


def fit_standardizers_from_paths(
    energy_paths,
    weather: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    max_interp_gap: int,
) -> StandardizationStats:
    """Fit feature and target scaling statistics on training splits only.

    :param energy_paths: Iterable of house CSV paths.
    :param weather: Preprocessed weather dataframe.
    :param train_frac: Fraction used for training split.
    :param val_frac: Fraction used for validation split.
    :param max_interp_gap: Maximum number of missing minutes to interpolate.
    :return: Standardization statistics for features and targets.
    """
    feature_rows = 0
    target_rows = 0
    feature_sum = np.zeros(len(INPUT_FEATURES), dtype=np.float64)
    feature_sum_sq = np.zeros(len(INPUT_FEATURES), dtype=np.float64)
    target_sum = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
    target_sum_sq = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)

    for path in tqdm(energy_paths, desc="Fitting scalers", leave=False):
        train_frame = split_house_frame(
            load_house_frame(path, weather, max_interp_gap),
            train_frac,
            val_frac,
        )["train"]
        feature_rows, feature_sum, feature_sum_sq = update_running_stats(
            train_frame,
            INPUT_FEATURES,
            feature_rows,
            feature_sum,
            feature_sum_sq,
        )
        target_rows, target_sum, target_sum_sq = update_running_stats(
            train_frame,
            TARGET_COLUMNS,
            target_rows,
            target_sum,
            target_sum_sq,
        )

    feature_mean, feature_std = finalize_running_stats(feature_rows, feature_sum, feature_sum_sq)
    target_mean, target_std = finalize_running_stats(target_rows, target_sum, target_sum_sq)
    return StandardizationStats(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
