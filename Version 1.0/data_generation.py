"""
data_generation.py
==================
Part 1: Dataset Generation for Character Recognition
Generates noisy 8x8 pixel templates for B, 0 (zero), and E.

Each pixel is either -1.0 (black) or +1.0 (white).
100 noisy versions of each character are created by adding
uniform noise in [-5.0, +5.0] to each pixel independently.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── 8×8 Pixel Templates ──────────────────────────────────────────────────────
# Convention: -1.0 = black (ink), +1.0 = white (background)
# Grids are defined row-by-row, top to bottom.

# Capital B  (8×8)
TEMPLATE_B = np.array([
    [-1, -1, -1, -1, -1,  1,  1,  1],   # row 0
    [-1,  1,  1,  1, -1,  1,  1,  1],   # row 1
    [-1,  1,  1,  1, -1,  1,  1,  1],   # row 2
    [-1, -1, -1, -1, -1,  1,  1,  1],   # row 3
    [-1,  1,  1,  1,  1, -1,  1,  1],   # row 4
    [-1,  1,  1,  1,  1,  1, -1,  1],   # row 5
    [-1,  1,  1,  1,  1,  1, -1,  1],   # row 6
    [-1, -1, -1, -1, -1, -1,  1,  1],   # row 7
], dtype=np.float64)

# Capital 0 / Zero  (8×8)
TEMPLATE_0 = np.array([
    [ 1,  1, -1, -1, -1, -1,  1,  1],   # row 0
    [ 1, -1,  1,  1,  1,  1, -1,  1],   # row 1
    [-1,  1,  1,  1,  1,  1,  1, -1],   # row 2
    [-1,  1,  1,  1,  1,  1,  1, -1],   # row 3
    [-1,  1,  1,  1,  1,  1,  1, -1],   # row 4
    [-1,  1,  1,  1,  1,  1,  1, -1],   # row 5
    [ 1, -1,  1,  1,  1,  1, -1,  1],   # row 6
    [ 1,  1, -1, -1, -1, -1,  1,  1],   # row 7
], dtype=np.float64)

# Capital E  (8×8)
TEMPLATE_E = np.array([
    [-1, -1, -1, -1, -1, -1, -1,  1],   # row 0
    [-1,  1,  1,  1,  1,  1,  1,  1],   # row 1
    [-1,  1,  1,  1,  1,  1,  1,  1],   # row 2
    [-1, -1, -1, -1, -1,  1,  1,  1],   # row 3
    [-1,  1,  1,  1,  1,  1,  1,  1],   # row 4
    [-1,  1,  1,  1,  1,  1,  1,  1],   # row 5
    [-1,  1,  1,  1,  1,  1,  1,  1],   # row 6
    [-1, -1, -1, -1, -1, -1, -1,  1],   # row 7
], dtype=np.float64)

# Map class index -> (template, one-hot label, name)
CLASS_INFO = {
    0: (TEMPLATE_B, np.array([1, 0, 0]), "B"),
    1: (TEMPLATE_0, np.array([0, 1, 0]), "0"),
    2: (TEMPLATE_E, np.array([0, 0, 1]), "E"),
}

NOISE_LOW  = -5.0
NOISE_HIGH = +5.0
SAMPLES_PER_CLASS = 100


def generate_noisy_samples(template: np.ndarray,
                            n_samples: int = SAMPLES_PER_CLASS,
                            noise_low: float = NOISE_LOW,
                            noise_high: float = NOISE_HIGH,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate `n_samples` noisy versions of a character template.

    Parameters
    ----------
    template   : (8, 8) array with values in {-1.0, +1.0}
    n_samples  : number of noisy copies to create
    noise_low  : lower bound of uniform noise
    noise_high : upper bound of uniform noise
    rng        : optional numpy random Generator for reproducibility

    Returns
    -------
    samples : (n_samples, 64) float array  — each row is one flattened sample
    """
    if rng is None:
        rng = np.random.default_rng()

    flat = template.flatten()                         # shape (64,)
    # Uniform noise: shape (n_samples, 64)
    noise = rng.uniform(noise_low, noise_high, size=(n_samples, flat.shape[0]))
    samples = flat[np.newaxis, :] + noise             # broadcast template + noise
    return samples


def generate_dataset(n_per_class: int = SAMPLES_PER_CLASS,
                     seed: int = 42) -> tuple:
    """
    Build the full 300-sample (or n_per_class × 3) dataset.

    Returns
    -------
    X : (N, 64)  float64  — pixel features
    y : (N, 3)   float64  — one-hot labels
    labels : (N,) int     — class indices 0/1/2
    """
    rng = np.random.default_rng(seed)
    X_list, y_list, label_list = [], [], []

    for class_idx, (template, one_hot, name) in CLASS_INFO.items():
        samples = generate_noisy_samples(template, n_samples=n_per_class, rng=rng)
        X_list.append(samples)
        y_list.append(np.tile(one_hot, (n_per_class, 1)))
        label_list.append(np.full(n_per_class, class_idx, dtype=int))
        print(f"  Class '{name}' ({class_idx}): {n_per_class} samples generated.")

    X = np.vstack(X_list)          # (N, 64)
    y = np.vstack(y_list)          # (N, 3)
    labels = np.concatenate(label_list)  # (N,)

    # Shuffle the dataset
    perm = rng.permutation(len(X))
    return X[perm], y[perm], labels[perm]


def save_dataset(X, y, labels, out_dir: str = "data"):
    """Persist the dataset to .npy files."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    print(f"\nDataset saved to '{out_dir}/'")


def load_dataset(out_dir: str = "data") -> tuple:
    """Load a previously saved dataset."""
    X      = np.load(os.path.join(out_dir, "X.npy"))
    y      = np.load(os.path.join(out_dir, "y.npy"))
    labels = np.load(os.path.join(out_dir, "labels.npy"))
    return X, y, labels


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_templates(save_path: str = None):
    """
    Display the three clean 8×8 templates side by side.
    Black pixels (-1) are shown dark; white pixels (+1) are shown light.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle("Clean 8×8 Character Templates", fontsize=14, fontweight="bold")

    for ax, (class_idx, (template, _, name)) in zip(axes, CLASS_INFO.items()):
        # imshow: vmin/vmax maps -1->dark, +1->light
        im = ax.imshow(template, cmap="gray", vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"Class '{name}'  (label {class_idx})", fontsize=12)
        ax.axis("off")

        # Draw pixel grid lines
        for x in range(9):
            ax.axvline(x - 0.5, color="lightgray", linewidth=0.5)
        for y in range(9):
            ax.axhline(y - 0.5, color="lightgray", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Templates plot saved -> {save_path}")
    plt.show()
    return fig


def plot_noisy_samples(X, labels, n_cols: int = 5, save_path: str = None):
    """
    Show a few noisy samples for each class to verify noise addition.
    """
    n_show = n_cols                         # samples per class to display
    n_classes = 3
    fig, axes = plt.subplots(n_classes, n_show,
                             figsize=(n_show * 1.8, n_classes * 2.0))
    fig.suptitle(f"Noisy Samples — {n_show} per class\n"
                 f"(Uniform noise ∈ [{NOISE_LOW}, {NOISE_HIGH}])",
                 fontsize=13, fontweight="bold")

    class_names = {0: "B", 1: "0", 2: "E"}
    vmin, vmax = X.min(), X.max()

    for class_idx in range(n_classes):
        idx = np.where(labels == class_idx)[0][:n_show]
        for col, i in enumerate(idx):
            ax = axes[class_idx][col]
            ax.imshow(X[i].reshape(8, 8), cmap="RdBu_r",
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"'{class_names[class_idx]}'",
                              fontsize=12, rotation=0,
                              labelpad=20, va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Noisy samples plot saved -> {save_path}")
    plt.show()
    return fig


def plot_pixel_distribution(X, labels, save_path: str = None):
    """
    Plot histogram of pixel intensity values to confirm noise range.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    fig.suptitle("Pixel Intensity Distribution per Class", fontsize=13,
                 fontweight="bold")

    class_names = {0: "B", 1: "0", 2: "E"}
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for class_idx in range(3):
        ax = axes[class_idx]
        pixels = X[labels == class_idx].flatten()
        ax.hist(pixels, bins=40, color=colors[class_idx],
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.set_title(f"Class '{class_names[class_idx]}'", fontsize=11)
        ax.set_xlabel("Pixel value", fontsize=9)
        ax.axvline(-1.0, color="gray", linestyle="--", linewidth=1,
                   label="Template pixels")
        ax.axvline(+1.0, color="gray", linestyle="--", linewidth=1)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Frequency", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Distribution plot saved -> {save_path}")
    plt.show()
    return fig


def print_dataset_summary(X, y, labels):
    """Print a concise summary of the generated dataset."""
    class_names = {0: "B", 1: "0", 2: "E"}
    print("=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total samples  : {len(X)}")
    print(f"  Feature dim    : {X.shape[1]}  (8×8 = 64 pixels)")
    print(f"  Label shape    : {y.shape}  (one-hot, 3 classes)")
    print(f"  Pixel range    : [{X.min():.2f}, {X.max():.2f}]")
    print()
    for ci in range(3):
        cnt = np.sum(labels == ci)
        print(f"  Class '{class_names[ci]}' (index {ci})  "
              f"->  {cnt} samples,  label = {y[labels == ci][0]}")
    print("=" * 50)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    print("\n[1] Generating dataset …")
    X, y, labels = generate_dataset(n_per_class=SAMPLES_PER_CLASS, seed=42)

    print_dataset_summary(X, y, labels)

    print("\n[2] Saving dataset …")
    save_dataset(X, y, labels, out_dir="data")

    print("\n[3] Plotting templates …")
    plot_templates(save_path="plots/templates.png")

    print("\n[4] Plotting noisy samples …")
    plot_noisy_samples(X, labels, n_cols=5,
                       save_path="plots/noisy_samples.png")

    print("\n[5] Plotting pixel distributions …")
    plot_pixel_distribution(X, labels,
                            save_path="plots/pixel_distribution.png")

    print("\nPart 1 complete.")
