"""Dataset utilities for sequence and tabular energy model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from storenet_ml.config import INPUT_FEATURES, TARGET_COLUMNS


@dataclass
class HouseSequence:
    """Normalized time-series arrays for a single house."""

    house_id: int
    features: np.ndarray
    targets: np.ndarray


class SlidingWindowDataset(Dataset):
    """PyTorch dataset that yields sliding windows from house sequences."""

    def __init__(
        self,
        sequences: list[HouseSequence],
        seq_len: int,
        horizon: int,
        stride: int,
    ) -> None:
        """Initialize a sliding-window dataset.

        :param sequences: House sequences to sample from.
        :param seq_len: Input window length in timesteps.
        :param horizon: Prediction offset from the end of each window.
        :param stride: Step size between window starts.
        """
        self.sequences = sequences
        self.seq_len = seq_len
        self.horizon = horizon
        self.stride = stride
        self.sample_counts = np.array(
            [self._sample_count(sequence.features.shape[0]) for sequence in sequences],
            dtype=np.int64,
        )
        self.cumulative = np.cumsum(self.sample_counts)

    def _sample_count(self, series_length: int) -> int:
        """Compute the number of valid windows for one sequence length.

        :param series_length: Number of timesteps in a sequence.
        :return: Number of extractable sliding-window samples.
        """
        usable = series_length - self.seq_len - self.horizon + 1
        if usable <= 0:
            return 0
        return ((usable - 1) // self.stride) + 1

    def __len__(self) -> int:
        """Return the total number of samples across all sequences.

        :return: Dataset length.
        """
        if len(self.cumulative) == 0:
            return 0
        return int(self.cumulative[-1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch one sample window, target row, and house identifier.

        :param index: Global sample index.
        :return: Tuple ``(x, y, house_id)`` as PyTorch tensors.
        """
        seq_index = int(np.searchsorted(self.cumulative, index, side="right"))
        prev_total = 0 if seq_index == 0 else int(self.cumulative[seq_index - 1])
        local_index = index - prev_total
        start = local_index * self.stride

        sequence = self.sequences[seq_index]
        stop = start + self.seq_len
        target_index = stop + self.horizon - 1

        x = torch.from_numpy(sequence.features[start:stop])
        y = torch.from_numpy(sequence.targets[target_index])
        house_id = torch.tensor(sequence.house_id, dtype=torch.long)
        return x, y, house_id


def split_contiguous_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a frame into contiguous 1-minute segments.

    :param frame: Input dataframe with a ``date`` timestamp column.
    :return: List of non-empty contiguous segments.
    """
    if frame.empty:
        return []

    frame = frame.sort_values("date").reset_index(drop=True)
    gap_start = frame["date"].diff().ne(pd.Timedelta(minutes=1))
    gap_start.iloc[0] = True
    segment_ids = gap_start.cumsum()
    return [
        segment.reset_index(drop=True)
        for _, segment in frame.groupby(segment_ids)
        if not segment.empty
    ]


def build_sequences_from_frame(
    frame: pd.DataFrame,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> list[HouseSequence]:
    """Create normalized house sequences from contiguous frame segments.

    :param frame: House dataframe to convert.
    :param feature_mean: Feature means used for normalization.
    :param feature_std: Feature standard deviations used for normalization.
    :param target_mean: Target means used for normalization.
    :param target_std: Target standard deviations used for normalization.
    :return: List of normalized ``HouseSequence`` objects.
    """
    sequences: list[HouseSequence] = []
    for segment in split_contiguous_segments(frame):
        feature_values = segment[INPUT_FEATURES].to_numpy(dtype=np.float32)
        target_values = segment[TARGET_COLUMNS].to_numpy(dtype=np.float32)

        normalized_features = (feature_values - feature_mean) / feature_std
        normalized_targets = (target_values - target_mean) / target_std

        sequences.append(
            HouseSequence(
                house_id=int(segment["house_id"].iloc[0]),
                features=normalized_features,
                targets=normalized_targets,
            )
        )

    return sequences


def build_tabular_examples_from_frame(
    frame: pd.DataFrame,
    seq_len: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flattened tabular windows and targets from a house frame.

    :param frame: House dataframe to convert.
    :param seq_len: Input window length in timesteps.
    :param horizon: Prediction offset from the end of each window.
    :param stride: Step size between window starts.
    :return: Tuple ``(features, targets)`` as float32 arrays.
    """
    # pylint: disable=too-many-locals
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []

    for segment in split_contiguous_segments(frame):
        if len(segment) < seq_len + horizon:
            continue

        feature_values = segment[INPUT_FEATURES].to_numpy(dtype=np.float32)
        target_values = segment[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        house_id = float(segment["house_id"].iloc[0])

        last_start = len(segment) - seq_len - horizon + 1
        for start in range(0, last_start, stride):
            stop = start + seq_len
            target_index = stop + horizon - 1
            window = feature_values[start:stop].reshape(-1)
            row = np.concatenate([window, np.array([house_id], dtype=np.float32)])
            feature_rows.append(row)
            target_rows.append(target_values[target_index])

    if not feature_rows:
        return (
            np.empty((0, seq_len * len(INPUT_FEATURES) + 1), dtype=np.float32),
            np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32),
        )

    return np.stack(feature_rows), np.stack(target_rows)


def create_dataloader(dataset: SlidingWindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a dataloader for a sliding-window dataset.

    :param dataset: Dataset to iterate.
    :param batch_size: Number of samples per batch.
    :param shuffle: Whether to shuffle samples each epoch.
    :return: Configured PyTorch dataloader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
    )
