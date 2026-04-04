"""
Dataset D: 300 training samples — 100 noisy copies per class (B, 0, E).
Noise: independent Uniform[-5, 5] per pixel (assignment specification).
"""
from __future__ import annotations

import numpy as np

from letter_patterns import get_base_patterns


def generate_dataset(
    n_per_class: int = 100,
    noise_low: float = -5.0,
    noise_high: float = 5.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: (300, 64) feature vectors (row-major flatten of 8x8)
        y: (300,) integer labels 0=B, 1=0, 2=E
    """
    rng = np.random.default_rng(seed)
    bases = get_base_patterns()
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, pattern in enumerate(bases):
        flat = pattern.reshape(64)
        for _ in range(n_per_class):
            # Independent uniform noise per pixel (assignment specification).
            noise = rng.uniform(noise_low, noise_high, size=64)
            xs.append(flat + noise)
            ys.append(label)
    X = np.stack(xs, axis=0).astype(np.float64)
    y = np.array(ys, dtype=np.int64)
    return X, y
