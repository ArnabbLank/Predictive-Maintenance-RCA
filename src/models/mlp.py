"""
models/mlp.py — Simple MLP baseline for tabular RUL prediction.
"""

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron for RUL regression from tabular features.

    Architecture:
        Input → FC(hidden1) → ReLU → Dropout → FC(hidden2) → ReLU → Dropout → FC(1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)

        Returns
        -------
        Tensor, shape (batch, 1)
        """
        return self.net(x)
