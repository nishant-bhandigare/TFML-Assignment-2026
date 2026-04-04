"""
Map uploaded RGB/grayscale images to 64-dim vectors consistent with training:
dark strokes ≈ -1, bright background ≈ +1 (same convention as letter_patterns).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def image_to_feature_vector(pil_image: Image.Image, size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Resize to 8×8, convert to grayscale, map pixel in [0,255] to [-1, +1]:
    black (0) -> -1, white (255) -> +1.
    Returns shape (64,) in row-major order.
    """
    gray = pil_image.convert("L").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float64) / 255.0
    # 0 -> -1, 1 -> +1
    feat = arr * 2.0 - 1.0
    return feat.reshape(64)


def apply_standardization(
    vec: np.ndarray,
    mu: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    """Match training pipeline when checkpoint stores train-split mu/std (shape (1,64))."""
    if mu is None or std is None:
        return vec
    return ((vec.reshape(1, -1) - mu) / std).reshape(64)
