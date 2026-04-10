"""Shared constants and utility helpers."""

import os
import random
from typing import Tuple

import numpy as np
import tensorflow as tf


CLASS_NAMES = ["B", "0", "E"]
INPUT_DIM = 64
N_CLASSES = 3


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def one_hot(labels: np.ndarray, num_classes: int = N_CLASSES) -> np.ndarray:
    """Convert class indices to one-hot vectors."""
    return np.eye(num_classes, dtype=np.float32)[labels]


def train_val_split_stratified(
    X: np.ndarray,
    y_idx: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified split preserving per-class ratio."""
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(y_idx):
        cls_idx = np.where(y_idx == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_frac))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    return X[train_idx], y_idx[train_idx], X[val_idx], y_idx[val_idx]
