#!/usr/bin/env python3
"""Estimate forward-pass FLOPs for the neural energy model variants."""

from __future__ import annotations

import json

import torch

from storenet_ml.config import ARTIFACT_DIR, HOUSE_ORDER, INPUT_FEATURES
from storenet_ml.models import SharedEnergyRNN, SharedEnergyTCN, SharedEnergyTransformer

FLOP_BATCH_SIZE = 1

# Best configurations from dataset_exploration/artifacts/grid_search_summary.json.
RNN_SEQ_LEN = 60
RNN_HIDDEN_SIZE = 128
RNN_HOUSE_EMBEDDING_DIM = 8
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.25

TCN_SEQ_LEN = 60
TCN_HIDDEN_SIZE = 128
TCN_HOUSE_EMBEDDING_DIM = 8
TCN_NUM_LAYERS = 8
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.15

TRANSFORMER_SEQ_LEN = 60
TRANSFORMER_D_MODEL = 64
TRANSFORMER_HOUSE_EMBEDDING_DIM = 8
TRANSFORMER_NUM_LAYERS = 8
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FEEDFORWARD_DIM = 96
TRANSFORMER_DROPOUT = 0.1


def format_human(value: int) -> str:
    """Format an integer quantity with engineering suffixes.

    :param value: Raw integer value.
    :return: Human-readable string using K/M/G/T/P suffixes.
    """
    suffixes = ["", "K", "M", "G", "T", "P"]
    v = float(value)
    for suffix in suffixes:
        if abs(v) < 1000.0:
            return f"{v:.2f}{suffix}"
        v /= 1000.0
    return f"{v:.2f}E"


def count_forward_flops(
    model: torch.nn.Module,
    x: torch.Tensor,
    house_ids: torch.Tensor,
) -> int:
    """Count forward-pass FLOPs for one model invocation.

    :param model: Model to profile.
    :param x: Input tensor for the forward pass.
    :param house_ids: House-id tensor paired with ``x``.
    :return: Total forward-pass FLOP count.
    :raises RuntimeError: If PyTorch FLOP counter utilities are unavailable.
    """
    # pylint: disable=import-outside-toplevel
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:
        raise RuntimeError(
            "PyTorch built-in FLOP counter is unavailable. "
            "Please use a PyTorch version with "
            "`torch.utils.flop_counter.FlopCounterMode` (2.0.0+)."
        ) from exc

    model.eval()
    with torch.no_grad():
        counter = FlopCounterMode(display=False)
        with counter:
            _ = model(x, house_ids)
        return int(counter.get_total_flops())


def build_inputs(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build synthetic tensors matching the model input contract.

    :param batch_size: Number of sequences in the synthetic batch.
    :param seq_len: Sequence length in timesteps.
    :param device: Device where tensors should be allocated.
    :return: Tuple ``(x, house_ids)``.
    """
    x = torch.randn(
        batch_size,
        seq_len,
        len(INPUT_FEATURES),
        device=device,
        dtype=torch.float32,
    )
    house_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
    return x, house_ids


def main() -> None:
    """Estimate forward-pass FLOPs for RNN, TCN, and transformer models."""
    device = torch.device("cpu")

    models = {
        "rnn": (
            SharedEnergyRNN(
                feature_dim=len(INPUT_FEATURES),
                num_houses=len(HOUSE_ORDER),
                hidden_size=RNN_HIDDEN_SIZE,
                house_embedding_dim=RNN_HOUSE_EMBEDDING_DIM,
                num_layers=RNN_NUM_LAYERS,
                dropout=RNN_DROPOUT,
            ).to(device),
            FLOP_BATCH_SIZE,
            RNN_SEQ_LEN,
        ),
        "tcn": (
            SharedEnergyTCN(
                feature_dim=len(INPUT_FEATURES),
                num_houses=len(HOUSE_ORDER),
                hidden_size=TCN_HIDDEN_SIZE,
                house_embedding_dim=TCN_HOUSE_EMBEDDING_DIM,
                num_layers=TCN_NUM_LAYERS,
                kernel_size=TCN_KERNEL_SIZE,
                dropout=TCN_DROPOUT,
            ).to(device),
            FLOP_BATCH_SIZE,
            TCN_SEQ_LEN,
        ),
        "transformer": (
            SharedEnergyTransformer(
                feature_dim=len(INPUT_FEATURES),
                num_houses=len(HOUSE_ORDER),
                max_seq_len=TRANSFORMER_SEQ_LEN,
                d_model=TRANSFORMER_D_MODEL,
                house_embedding_dim=TRANSFORMER_HOUSE_EMBEDDING_DIM,
                num_layers=TRANSFORMER_NUM_LAYERS,
                num_heads=TRANSFORMER_NUM_HEADS,
                feedforward_dim=TRANSFORMER_FEEDFORWARD_DIM,
                dropout=TRANSFORMER_DROPOUT,
            ).to(device),
            FLOP_BATCH_SIZE,
            TRANSFORMER_SEQ_LEN,
        ),
    }

    summary = {}
    for name, (model, batch_size, seq_len) in models.items():
        x, house_ids = build_inputs(batch_size=batch_size, seq_len=seq_len, device=device)
        forward_flops = count_forward_flops(model, x, house_ids)
        summary[name] = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "feature_dim": len(INPUT_FEATURES),
            "forward_flops": forward_flops,
            "forward_flops_human": format_human(forward_flops),
        }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    output_path = ARTIFACT_DIR / "neural_flops.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2))
    print("Saved FLOP summary to:", output_path)


if __name__ == "__main__":
    main()
