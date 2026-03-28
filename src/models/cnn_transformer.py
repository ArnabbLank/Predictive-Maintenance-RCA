"""
models/cnn_transformer.py — CNN-Transformer with MC Dropout (Paper B inspired).

Paper B: "Uncertainty Quantification and Remaining Useful Life Prediction
Using Deep Learning" (IJSIMM 2025)

Architecture:
    1D-CNN feature extractor
    → Positional Encoding
    → Transformer Encoder (multi-head self-attention)
    → Global Average Pooling
    → FC regression head with Dropout (kept active at inference for MC Dropout)
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CNNTransformerModel(nn.Module):
    """
    CNN-Transformer for RUL prediction with MC Dropout support.

    Input shape:  (batch, seq_len, n_features)
    Output shape: (batch, 1)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 30,
        cnn_channels: int = 64,
        kernel_size: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        mc_dropout: float = 0.1,
    ):
        super().__init__()
        self.mc_dropout_rate = mc_dropout

        # 1D CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, d_model, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # MC Dropout (stays active during inference)
        self.mc_dropout = nn.Dropout(mc_dropout)

        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(mc_dropout),  # MC dropout in head too
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass (point estimate).

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)
        """
        # CNN: (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, d_model)

        # Positional encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)

        # MC Dropout + FC
        x = self.mc_dropout(x)
        return self.fc(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout inference for uncertainty quantification.

        Performs `n_samples` stochastic forward passes with dropout
        active, then computes statistics across predictions.

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, n_features)
        n_samples : int
            Number of MC forward passes. Paper B uses 50–100.

        Returns
        -------
        mean : np.ndarray, shape (batch,)
            Mean RUL prediction.
        std : np.ndarray, shape (batch,)
            Standard deviation (epistemic uncertainty).
        all_preds : np.ndarray, shape (n_samples, batch)
            All individual predictions.
        """
        self.train()  # Keep dropout active!
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x).squeeze(-1)
                predictions.append(pred.cpu().numpy())

        all_preds = np.array(predictions)  # (n_samples, batch)
        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)

        return mean, std, all_preds
