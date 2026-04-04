"""
(1.b) Input→hidden weights as 8×8 images per hidden unit.
(1.c) Hidden→output weights: heatmap (outputs × hidden units) + per-output bar charts.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from letter_patterns import CLASS_NAMES
from model import LetterMLP


def plot_input_hidden_weights(
    model: LetterMLP,
    out_dir: str,
    prefix: str = "weights",
) -> str:
    """Each row of W1 (after fc1) is incoming weights to one hidden unit; reshape to 8×8."""
    os.makedirs(out_dir, exist_ok=True)
    # fc1.weight: (hidden, 64)
    w = model.fc1.weight.detach().cpu().numpy()
    h = w.shape[0]
    ncols = min(4, h)
    nrows = int(np.ceil(h / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i in range(h):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        img = w[i].reshape(8, 8)
        scale = float(np.max(np.abs(w))) or 1e-8
        im = ax.imshow(img, cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set_title(f"Hidden unit {i}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for j in range(h, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    plt.suptitle(
        "Input→hidden weights (8×8): each unit’s receptive field over the pixel grid",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_input_hidden_8x8.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_hidden_output_weights(
    model: LetterMLP,
    out_dir: str,
    prefix: str = "weights",
) -> tuple[str, str]:
    """fc2.weight shape (3, hidden): rows = output logits for B, 0, E."""
    os.makedirs(out_dir, exist_ok=True)
    w = model.fc2.weight.detach().cpu().numpy()
    b = model.fc2.bias.detach().cpu().numpy()
    hidden = w.shape[1]

    fig, ax = plt.subplots(figsize=(max(6, hidden * 0.8), 4))
    im = ax.imshow(w, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(hidden))
    ax.set_xticklabels([f"h{j}" for j in range(hidden)])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"out: {CLASS_NAMES[k]}" for k in range(3)])
    ax.set_xlabel("Hidden unit index")
    ax.set_ylabel("Output (class logit weights)")
    for i in range(3):
        for j in range(hidden):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Weight")
    plt.title("Hidden→output weight matrix W2 (rows = classes B, 0, E)")
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, f"{prefix}_hidden_output_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    fig2, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    x = np.arange(hidden)
    for k in range(3):
        axes[k].bar(x, w[k], color=["#2c7bb6", "#fdae61", "#d7191c"][k])
        axes[k].set_title(f"Output {CLASS_NAMES[k]} (bias={b[k]:.3f})")
        axes[k].set_xlabel("Hidden unit")
        axes[k].set_xticks(x)
    axes[0].set_ylabel("Weight to hidden units")
    plt.suptitle("Per-class hidden→output weights (how each output combines hidden features)")
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"{prefix}_hidden_output_bars.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()

    return heatmap_path, bar_path
