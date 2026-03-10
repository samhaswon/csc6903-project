#!/usr/bin/env python3
"""Train and evaluate a shared LSTM model for house-level energy targets."""

from __future__ import annotations

import json

import torch
from torch import nn
from tqdm.auto import tqdm

from storenet_ml.config import ARTIFACT_DIR, HOUSE_ORDER, INPUT_FEATURES
from storenet_ml.datasets import create_dataloader
from storenet_ml.models import SharedEnergyRNN
from storenet_ml.pipelines import build_rnn_datasets
from storenet_ml.training import (
    collect_predictions,
    compute_metrics,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


# Training configuration
SEQ_LEN = 60
HORIZON = 1
STRIDE = 15
BATCH_SIZE = 256
HIDDEN_SIZE = 128
HOUSE_EMBEDDING_DIM = 16
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
MAX_INTERP_GAP = 5
SEED = 42


def build_config() -> dict:
    """Assemble the training configuration dictionary.

    :return: Dictionary of data, model, and optimization settings.
    """
    return {
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "house_embedding_dim": HOUSE_EMBEDDING_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "max_interp_gap": MAX_INTERP_GAP,
        "seed": SEED,
    }


def main() -> None:
    """Train and evaluate the shared RNN energy model."""
    # pylint: disable=too-many-locals
    config = build_config()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training script, "
            "but no CUDA device is available."
        )

    device = torch.device("cuda")
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset, stats = build_rnn_datasets(
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=STRIDE,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        max_interp_gap=MAX_INTERP_GAP,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(
            "At least one split has zero windows. Reduce SEQ_LEN/HORIZON or adjust split fractions."
        )

    print(
        "Window counts:",
        {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    )

    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SharedEnergyRNN(
        feature_dim=len(INPUT_FEATURES),
        num_houses=len(HOUSE_ORDER),
        hidden_size=HIDDEN_SIZE,
        house_embedding_dim=HOUSE_EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_checkpoint = None
    patience_left = PATIENCE

    epoch_progress = tqdm(range(1, EPOCHS + 1), desc="Training")
    for epoch in epoch_progress:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            EPOCHS,
        )
        val_loss, val_predictions, val_targets = collect_predictions(
            model,
            val_loader,
            loss_fn,
            device,
            stats.target_mean,
            stats.target_std,
            desc=f"Epoch {epoch}/{EPOCHS} [val]",
        )
        val_metrics = compute_metrics(val_predictions, val_targets)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_joint_mae={val_metrics['joint_mae']:.4f} | "
            f"val_joint_rmse={val_metrics['joint_rmse']:.4f}"
        )
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_mae=f"{val_metrics['joint_mae']:.4f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = PATIENCE
            best_checkpoint = save_checkpoint(
                model,
                config,
                stats.feature_mean,
                stats.feature_std,
                stats.target_mean,
                stats.target_std,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    if best_checkpoint is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_predictions, test_targets = collect_predictions(
        model,
        test_loader,
        loss_fn,
        device,
        stats.target_mean,
        stats.target_std,
        desc="Testing",
    )
    test_metrics = compute_metrics(test_predictions, test_targets)
    test_metrics["test_loss"] = float(test_loss)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    metrics_path = ARTIFACT_DIR / "shared_energy_rnn_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print("Best checkpoint:", best_checkpoint)
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
