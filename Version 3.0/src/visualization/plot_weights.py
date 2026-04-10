"""Plot input-hidden and hidden-output weights for interpretation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.utils.helpers import CLASS_NAMES


def plot_weights(model_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    w1 = model.get_layer("hidden").get_weights()[0]  # (64, H)
    w2 = model.get_layer("output").get_weights()[0]  # (H, 3)

    # Input -> Hidden maps
    H = w1.shape[1]
    ncols = min(8, H)
    nrows = int(np.ceil(H / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i < H:
            ax.imshow(w1[:, i].reshape(8, 8), cmap="coolwarm")
            ax.set_title(f"H{i+1}")
            ax.axis("off")
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "input_hidden_weights_v3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Hidden -> Output bars
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for j, name in enumerate(CLASS_NAMES):
        axes[j].bar(np.arange(H), w2[:, j])
        axes[j].set_title(f"Output {name}")
        axes[j].set_xlabel("Hidden unit")
        axes[j].set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(out_dir / "hidden_output_weights_v3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    plot_weights(root / "models" / "trained_model.keras", root / "reports" / "figures")
