"""End-to-end data pipeline assembly for StoreNet model training."""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from storenet_ml.config import DATA_DIR
from storenet_ml.data_loaders import (
    fit_standardizers_from_paths,
    load_house_frame,
    load_weather,
    split_house_frame,
)
from storenet_ml.datasets import (
    SlidingWindowDataset,
    build_sequences_from_frame,
    build_tabular_examples_from_frame,
)


def list_energy_paths():
    """List house energy CSV paths sorted by numeric house id.

    :return: Sorted list of ``H*_Wh.csv`` paths.
    :raises FileNotFoundError: If no matching energy files are found.
    """
    energy_paths = sorted(
        DATA_DIR.glob("H*_Wh.csv"),
        key=lambda path: int(path.stem.split("_")[0][1:]),
    )
    if not energy_paths:
        raise FileNotFoundError(f"No energy files matching H*_Wh.csv found in {DATA_DIR}")
    return energy_paths


def build_rnn_datasets(
    seq_len: int,
    horizon: int,
    stride: int,
    train_frac: float,
    val_frac: float,
    max_interp_gap: int,
):
    """Build train/val/test sliding-window datasets and normalization stats.

    :param seq_len: Input window length in timesteps.
    :param horizon: Prediction offset from window end.
    :param stride: Step size between window starts.
    :param train_frac: Fraction used for training split.
    :param val_frac: Fraction used for validation split.
    :param max_interp_gap: Maximum number of missing minutes to interpolate.
    :return: Tuple ``(train_dataset, val_dataset, test_dataset, stats)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    weather = load_weather(max_interp_gap)
    energy_paths = list_energy_paths()
    stats = fit_standardizers_from_paths(
        energy_paths,
        weather,
        train_frac=train_frac,
        val_frac=val_frac,
        max_interp_gap=max_interp_gap,
    )

    train_sequences = []
    val_sequences = []
    test_sequences = []

    for path in tqdm(energy_paths, desc="Building sequences", leave=False):
        splits = split_house_frame(
            load_house_frame(path, weather, max_interp_gap),
            train_frac=train_frac,
            val_frac=val_frac,
        )
        train_sequences.extend(
            build_sequences_from_frame(
                splits["train"],
                stats.feature_mean,
                stats.feature_std,
                stats.target_mean,
                stats.target_std,
            )
        )
        val_sequences.extend(
            build_sequences_from_frame(
                splits["val"],
                stats.feature_mean,
                stats.feature_std,
                stats.target_mean,
                stats.target_std,
            )
        )
        test_sequences.extend(
            build_sequences_from_frame(
                splits["test"],
                stats.feature_mean,
                stats.feature_std,
                stats.target_mean,
                stats.target_std,
            )
        )

    train_dataset = SlidingWindowDataset(train_sequences, seq_len, horizon, stride)
    val_dataset = SlidingWindowDataset(val_sequences, seq_len, horizon, stride)
    test_dataset = SlidingWindowDataset(test_sequences, seq_len, horizon, stride)
    return train_dataset, val_dataset, test_dataset, stats


def build_tabular_splits(
    seq_len: int,
    horizon: int,
    stride: int,
    train_frac: float,
    val_frac: float,
    max_interp_gap: int,
):
    """Build flattened train/val/test arrays for tree-based models.

    :param seq_len: Input window length in timesteps.
    :param horizon: Prediction offset from window end.
    :param stride: Step size between window starts.
    :param train_frac: Fraction used for training split.
    :param val_frac: Fraction used for validation split.
    :param max_interp_gap: Maximum number of missing minutes to interpolate.
    :return: Tuple ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
    :raises RuntimeError: If any split has no tabular examples.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    weather = load_weather(max_interp_gap)
    energy_paths = list_energy_paths()

    train_features = []
    train_targets = []
    val_features = []
    val_targets = []
    test_features = []
    test_targets = []

    for path in tqdm(energy_paths, desc="Building tabular data", leave=False):
        splits = split_house_frame(
            load_house_frame(path, weather, max_interp_gap),
            train_frac=train_frac,
            val_frac=val_frac,
        )
        x_train, y_train = build_tabular_examples_from_frame(
            splits["train"],
            seq_len,
            horizon,
            stride,
        )
        x_val, y_val = build_tabular_examples_from_frame(
            splits["val"],
            seq_len,
            horizon,
            stride,
        )
        x_test, y_test = build_tabular_examples_from_frame(
            splits["test"],
            seq_len,
            horizon,
            stride,
        )

        if len(x_train):
            train_features.append(x_train)
            train_targets.append(y_train)
        if len(x_val):
            val_features.append(x_val)
            val_targets.append(y_val)
        if len(x_test):
            test_features.append(x_test)
            test_targets.append(y_test)

    def combine(parts):
        """Concatenate arrays in ``parts`` or return ``None`` when empty.

        :param parts: List of numpy arrays for one split.
        :return: Concatenated array or ``None``.
        """
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    x_train = combine(train_features)
    y_train = combine(train_targets)
    x_val = combine(val_features)
    y_val = combine(val_targets)
    x_test = combine(test_features)
    y_test = combine(test_targets)

    if x_train is None or x_val is None or x_test is None:
        raise RuntimeError(
            "At least one split has zero tabular examples. "
            "Reduce seq_len/horizon or adjust splits."
        )

    return x_train, y_train, x_val, y_val, x_test, y_test
