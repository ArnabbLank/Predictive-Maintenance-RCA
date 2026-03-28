"""
models/lstm.py — LSTM model for sequence-based RUL prediction.

Input shape:  (batch, seq_len, n_features)
Output shape: (batch, 1)
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    A multi-layer LSTM for RUL regression on windowed sensor data.

    Architecture:
        LSTM(n_layers) → take last hidden state → FC → ReLU → Dropout → FC(1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_size * self.n_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        Tensor, shape (batch, 1)
        """
        # lstm_out: (batch, seq_len, hidden*directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last time step output
        last_output = lstm_out[:, -1, :]  # (batch, hidden*directions)

        return self.fc(last_output)
