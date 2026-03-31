"""
models/cnn_lstm.py — CNN + LSTM model (Paper A inspired).

Paper A: "A method for predicting remaining useful life using enhanced
Savitzky-Golay filter and improved deep learning framework"

Architecture:
    1D-CNN blocks → LSTM → FC head

The CNN extracts local temporal features, then the LSTM captures
long-range temporal dependencies.
"""

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Single 1D-CNN block: Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class CNNLSTMModel(nn.Module):
    """
    CNN + LSTM hybrid for RUL prediction.

    Input shape:  (batch, seq_len, n_features)
    Output shape: (batch, 1)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 30,
        cnn_channels: list = None,
        kernel_size: int = 3,
        pool_size: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]

        # CNN blocks
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers.append(CNNBlock(in_ch, out_ch, kernel_size, pool_size))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute output sequence length after CNN pooling
        dummy_len = seq_len
        for _ in cnn_channels:
            dummy_len = dummy_len // pool_size
        self.cnn_out_len = max(dummy_len, 1)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # Back to (batch, seq_len', channels)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]

        return self.fc(last_out)
