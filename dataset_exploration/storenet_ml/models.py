from __future__ import annotations

import torch
from torch import nn

from storenet_ml.config import TARGET_COLUMNS


class SharedEnergyRNN(nn.Module):
    """House-aware LSTM forecaster for production and consumption targets."""

    def __init__(
        self,
        feature_dim: int,
        num_houses: int,
        hidden_size: int,
        house_embedding_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the shared RNN model.

        :param feature_dim: Number of per-timestep input features.
        :param num_houses: Number of unique houses for embeddings.
        :param hidden_size: Hidden dimension of the LSTM and MLP head.
        :param house_embedding_dim: Embedding dimension for house identifiers.
        :param num_layers: Number of stacked LSTM layers.
        :param dropout: Dropout probability used in recurrent/head layers.
        """
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.house_embedding = nn.Embedding(num_houses, house_embedding_dim)
        self.rnn = nn.LSTM(
            input_size=feature_dim + house_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(TARGET_COLUMNS)),
        )

    def forward(self, x: torch.Tensor, house_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass for one batch.

        :param x: Input tensor of shape ``(batch, seq_len, feature_dim)``.
        :param house_ids: House-id tensor of shape ``(batch,)``.
        :return: Predicted targets of shape ``(batch, len(TARGET_COLUMNS))``.
        """
        house_embedding = self.house_embedding(house_ids)
        repeated_embedding = house_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        rnn_input = torch.cat([x, repeated_embedding], dim=-1)
        encoded, _ = self.rnn(rnn_input)
        return self.head(encoded[:, -1, :])


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block with two Conv1D layers."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        """Initialize a temporal residual block.

        :param channels: Number of input and output channels.
        :param kernel_size: Kernel size for temporal convolutions.
        :param dilation: Dilation factor for both convolutions.
        :param dropout: Dropout probability after activations.
        """
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual temporal convolutions.

        :param x: Input tensor of shape ``(batch, channels, seq_len)``.
        :return: Tensor with the same shape after residual update.
        """
        return x + self.net(x)


class SharedEnergyTCN(nn.Module):
    """House-aware temporal convolutional network forecaster."""

    def __init__(
        self,
        feature_dim: int,
        num_houses: int,
        hidden_size: int,
        house_embedding_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        """Initialize the shared TCN model.

        :param feature_dim: Number of per-timestep input features.
        :param num_houses: Number of unique houses for embeddings.
        :param hidden_size: Channel count in the temporal stack.
        :param house_embedding_dim: Embedding dimension for house identifiers.
        :param num_layers: Number of residual temporal blocks.
        :param kernel_size: Kernel size used in temporal blocks.
        :param dropout: Dropout probability used in the network.
        """
        super().__init__()
        self.house_embedding = nn.Embedding(num_houses, house_embedding_dim)
        input_dim = feature_dim + house_embedding_dim
        self.input_projection = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        self.temporal_stack = nn.Sequential(
            *[
                TemporalConvBlock(
                    channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**layer_index,
                    dropout=dropout,
                )
                for layer_index in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(TARGET_COLUMNS)),
        )

    def forward(self, x: torch.Tensor, house_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass for one batch.

        :param x: Input tensor of shape ``(batch, seq_len, feature_dim)``.
        :param house_ids: House-id tensor of shape ``(batch,)``.
        :return: Predicted targets of shape ``(batch, len(TARGET_COLUMNS))``.
        """
        house_embedding = self.house_embedding(house_ids)
        repeated_embedding = house_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        features = torch.cat([x, repeated_embedding], dim=-1).transpose(1, 2)
        encoded = self.input_projection(features)
        encoded = self.temporal_stack(encoded)
        return self.head(encoded[:, :, -1])


class SharedEnergyTransformer(nn.Module):
    """House-aware causal transformer forecaster."""

    def __init__(
        self,
        feature_dim: int,
        num_houses: int,
        max_seq_len: int,
        d_model: int,
        house_embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        """Initialize the shared transformer model.

        :param feature_dim: Number of per-timestep input features.
        :param num_houses: Number of unique houses for embeddings.
        :param max_seq_len: Maximum supported sequence length.
        :param d_model: Transformer token embedding width.
        :param house_embedding_dim: Embedding dimension for house identifiers.
        :param num_layers: Number of transformer encoder layers.
        :param num_heads: Number of attention heads per layer.
        :param feedforward_dim: Hidden size of feedforward sublayers.
        :param dropout: Dropout probability used in transformer/head layers.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.house_embedding = nn.Embedding(num_houses, house_embedding_dim)
        self.input_projection = nn.Linear(feature_dim + house_embedding_dim, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(TARGET_COLUMNS)),
        )

    def forward(self, x: torch.Tensor, house_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass for one batch.

        :param x: Input tensor of shape ``(batch, seq_len, feature_dim)``.
        :param house_ids: House-id tensor of shape ``(batch,)``.
        :return: Predicted targets of shape ``(batch, len(TARGET_COLUMNS))``.
        :raises ValueError: If ``seq_len`` exceeds ``max_seq_len``.
        """
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds configured max_seq_len={self.max_seq_len}.")

        house_embedding = self.house_embedding(house_ids)
        repeated_embedding = house_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        tokens = torch.cat([x, repeated_embedding], dim=-1)
        encoded = self.input_projection(tokens)

        positions = torch.arange(x.size(1), device=x.device)
        encoded = encoded + self.position_embedding(positions).unsqueeze(0)

        causal_mask = torch.triu(
            torch.full((x.size(1), x.size(1)), float("-inf"), device=x.device),
            diagonal=1,
        )
        encoded = self.transformer(encoded, mask=causal_mask, is_causal=True)
        encoded = self.norm(encoded)
        return self.head(encoded[:, -1, :])
