# cocp/models.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, num_hidden: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, num_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_hidden

        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class MeanNet(nn.Module):
    def __init__(self, input_dim: int, num_hidden: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(input_dim, 1, num_hidden, num_layers, dropout)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class ThresholdNet(nn.Module):
    def __init__(self, input_dim: int, num_hidden: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(input_dim, 1, num_hidden, num_layers, dropout)

    def forward(self, x):
        return F.softplus(self.mlp(x).squeeze(-1))