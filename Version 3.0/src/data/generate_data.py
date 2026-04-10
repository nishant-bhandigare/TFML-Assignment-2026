"""Generate assignment dataset for B, 0, E with uniform noise."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils.helpers import CLASS_NAMES, one_hot, set_global_seed

NOISE_LOW = -5.0
NOISE_HIGH = 5.0


def create_base_patterns() -> Dict[str, np.ndarray]:
    """Create 8x8 templates with -1 (black) and +1 (white)."""
    B = np.array([
        [-1, -1, -1, -1, -1,  1,  1,  1],
        [-1,  1,  1,  1, -1,  1,  1,  1],
        [-1,  1,  1,  1, -1,  1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1,  1],
        [-1,  1,  1,  1,  1, -1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1, -1, -1, -1, -1, -1,  1,  1],
    ], dtype=np.float32)

    O = np.array([
        [ 1,  1, -1, -1, -1, -1,  1,  1],
        [ 1, -1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1, -1],
        [ 1, -1,  1,  1,  1,  1, -1,  1],
        [ 1,  1, -1, -1, -1, -1,  1,  1],
    ], dtype=np.float32)

    E = np.array([
        [-1, -1, -1, -1, -1, -1, -1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1, -1, -1,  1],
    ], dtype=np.float32)

    return {"B": B, "0": O, "E": E}


def generate_dataset(samples_per_class: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate flattened dataset and integer/one-hot labels."""
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    patterns = create_base_patterns()
    X_list, y_idx_list = [], []

    for class_idx, name in enumerate(CLASS_NAMES):
        template = patterns[name].reshape(1, 64)
        noise = rng.uniform(NOISE_LOW, NOISE_HIGH, size=(samples_per_class, 64)).astype(np.float32)
        noisy_samples = template + noise
        X_list.append(noisy_samples)
        y_idx_list.append(np.full(samples_per_class, class_idx, dtype=np.int64))

    X = np.vstack(X_list).astype(np.float32)
    y_idx = np.concatenate(y_idx_list)
    y_onehot = one_hot(y_idx)

    perm = rng.permutation(len(X))
    return X[perm], y_idx[perm], y_onehot[perm]


def save_dataset(X: np.ndarray, y_idx: np.ndarray, y_onehot: np.ndarray, out_dir: Path) -> None:
    """Save dataset files for reproducible experiments."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y_idx.npy", y_idx)
    np.save(out_dir / "y_onehot.npy", y_onehot)


def plot_templates(patterns: Dict[str, np.ndarray], out_path: Path) -> None:
    """Save template visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, name in enumerate(CLASS_NAMES):
        axes[i].imshow(patterns[name], cmap="gray", vmin=-1, vmax=1)
        axes[i].set_title(f"Template {name}")
        axes[i].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    X, y_idx, y_onehot = generate_dataset(samples_per_class=100, seed=42)
    save_dataset(X, y_idx, y_onehot, root / "data" / "processed")
    plot_templates(create_base_patterns(), root / "reports" / "figures" / "templates_v3.png")
    print("Dataset generated:", X.shape, y_idx.shape, y_onehot.shape)
