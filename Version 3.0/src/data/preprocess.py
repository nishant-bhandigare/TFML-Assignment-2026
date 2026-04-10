"""Preprocessing helpers for training and web inference."""

from typing import Tuple

import numpy as np


def standardize_train_val(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Standardize using train-set statistics only."""
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-8)
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def standardize_with_stats(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize using precomputed statistics."""
    return (X - mean) / (std + 1e-8)
