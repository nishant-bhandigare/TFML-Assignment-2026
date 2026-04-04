"""
64 — X — 3 feedforward network with biases (one hidden layer).
Supports optional LayerNorm, dropout, and multiple activations for better generalization.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n == "tanh":
        return nn.Tanh()
    if n == "silu" or n == "swish":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class LetterMLP(nn.Module):
    """Fully connected network: input_dim -> hidden_dim -> num_classes."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 3,
        num_classes: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.use_layer_norm = use_layer_norm
        self.ln = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.ln is not None:
            x = self.ln(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)

    @property
    def hidden_dim(self) -> int:
        return self.fc1.out_features


def letter_mlp_from_checkpoint(ckpt: dict) -> LetterMLP:
    """
    Load architecture compatible with saved weights.
    version < 2: older checkpoints (ReLU, no LayerNorm, no dropout).
    """
    h = int(ckpt.get("hidden_dim", 3))
    ver = int(ckpt.get("version", 1))
    if ver < 2:
        return LetterMLP(hidden_dim=h, dropout=0.0, use_layer_norm=False, activation="relu")
    return LetterMLP(
        hidden_dim=h,
        dropout=float(ckpt.get("dropout", 0.0)),
        use_layer_norm=bool(ckpt.get("use_layer_norm", True)),
        activation=str(ckpt.get("activation", "gelu")),
    )
