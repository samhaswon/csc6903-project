"""Training loops, evaluation, and checkpoint helpers for neural models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from storenet_ml.config import HOUSE_ORDER, INPUT_FEATURES, MODEL_DIR, TARGET_COLUMNS


def set_seed(seed: int) -> None:
    """Set NumPy and PyTorch random seeds.

    :param seed: Seed value used for RNG initialization.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train one epoch and return mean loss.

    :param model: Model to optimize.
    :param loader: Training dataloader.
    :param optimizer: Optimizer instance.
    :param loss_fn: Loss function.
    :param device: Device used for computation.
    :param epoch: One-based epoch index.
    :param total_epochs: Total epoch count for progress display.
    :return: Mean training loss for the epoch.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    model.train()
    running_loss = 0.0
    sample_count = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for x, y, house_ids in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        house_ids = house_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(x, house_ids)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(sample_count, 1)


def collect_predictions(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate a model and collect denormalized predictions and targets.

    :param model: Model to evaluate.
    :param loader: Evaluation dataloader.
    :param loss_fn: Loss function used for reporting.
    :param device: Device used for computation.
    :param target_mean: Target means used for denormalization.
    :param target_std: Target standard deviations used for denormalization.
    :param desc: Progress-bar label.
    :return: Tuple ``(mean_loss, predictions, targets)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    model.eval()
    running_loss = 0.0
    sample_count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    mean_tensor = torch.tensor(target_mean, device=device)
    std_tensor = torch.tensor(target_std, device=device)

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)
        for x, y, house_ids in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            house_ids = house_ids.to(device, non_blocking=True)

            batch_predictions = model(x, house_ids)
            loss = loss_fn(batch_predictions, y)

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

            predictions.append((batch_predictions * std_tensor + mean_tensor).cpu().numpy())
            targets.append((y * std_tensor + mean_tensor).cpu().numpy())

    all_predictions = np.concatenate(predictions, axis=0)
    all_targets = np.concatenate(targets, axis=0)
    return running_loss / max(sample_count, 1), all_predictions, all_targets


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for both target columns.

    :param predictions: Predicted target values.
    :param targets: Ground-truth target values.
    :return: Dictionary of per-target and joint MAE/RMSE/R2 metrics.
    """
    errors = predictions - targets
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))

    target_mean = np.mean(targets, axis=0, keepdims=True)
    ss_res = np.sum(np.square(errors), axis=0)
    ss_tot = np.sum(np.square(targets - target_mean), axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - (ss_res / ss_tot), 0.0)

    return {
        "production_mae": float(mae[0]),
        "consumption_mae": float(mae[1]),
        "production_rmse": float(rmse[0]),
        "consumption_rmse": float(rmse[1]),
        "production_r2": float(r2[0]),
        "consumption_r2": float(r2[1]),
        "joint_mae": float(np.mean(mae)),
        "joint_rmse": float(np.mean(rmse)),
    }


def save_checkpoint(
    model: nn.Module,
    config: dict,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    checkpoint_name: str = "shared_energy_rnn.pt",
) -> Path:
    """Save a model checkpoint with configuration and scaler metadata.

    :param model: Trained model to serialize.
    :param config: Training and model configuration values.
    :param feature_mean: Feature means used during training.
    :param feature_std: Feature standard deviations used during training.
    :param target_mean: Target means used during training.
    :param target_std: Target standard deviations used during training.
    :param checkpoint_name: Filename for the checkpoint.
    :return: Full checkpoint path.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    MODEL_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODEL_DIR / checkpoint_name
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_columns": INPUT_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "house_order": HOUSE_ORDER,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path
