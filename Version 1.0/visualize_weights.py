"""
visualize_weights.py
====================
Part 3: Weight Visualisation for the trained 64-3-3 network.

(a) Input -> Hidden weights: each hidden unit's 64 weights reshaped to 8×8 heatmap
(b) Hidden -> Output weights: bar charts showing how output units combine hidden activations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import os, sys
sys.path.insert(0, ".")
from model import NeuralNetwork


CLASS_NAMES   = ["B", "0", "E"]
HIDDEN_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


# ─── (a) Input -> Hidden Weight Heatmaps ──────────────────────────────────────

def plot_input_hidden_weights(net: NeuralNetwork, save_path: str = None):
    """
    For each of the H hidden units, reshape its 64 input weights into an 8×8
    image and display as a diverging heatmap (blue=negative, red=positive).

    A positive weight in a pixel position means that pixel strongly excites
    the hidden unit; negative means it suppresses it.
    """
    W1 = net.get_weights()["W1"]   # shape (64, H)
    H  = W1.shape[1]

    abs_max = np.abs(W1).max()
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig = plt.figure(figsize=(4 * H + 1, 5.5))
    fig.suptitle(
        "Part 3(a) — Input -> Hidden Weights  (reshaped to 8×8)\n"
        "Blue = negative weight (suppresses unit)   |   Red = positive weight (excites unit)",
        fontsize=12, fontweight="bold", y=1.01
    )

    for h in range(H):
        ax = fig.add_subplot(1, H, h + 1)
        w_img = W1[:, h].reshape(8, 8)
        im = ax.imshow(w_img, cmap="RdBu_r", norm=norm, interpolation="nearest")

        ax.set_title(f"Hidden Unit {h + 1}", fontsize=12, fontweight="bold",
                     color=HIDDEN_COLORS[h % len(HIDDEN_COLORS)])
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(range(8), fontsize=7)
        ax.set_yticklabels(range(8), fontsize=7)
        ax.tick_params(length=2)

        for x in range(9):
            ax.axvline(x - 0.5, color="gray", linewidth=0.3, alpha=0.5)
        for y in range(9):
            ax.axhline(y - 0.5, color="gray", linewidth=0.3, alpha=0.5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Weight value")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_input_hidden_overlay(net: NeuralNetwork, save_path: str = None):
    """
    Enhanced version: shows templates + hidden weight maps together
    so the reader can see which character features each unit responds to.
    """
    from data_generation import TEMPLATE_B, TEMPLATE_0, TEMPLATE_E
    templates = [TEMPLATE_B, TEMPLATE_0, TEMPLATE_E]

    W1 = net.get_weights()["W1"]   # (64, H)
    H  = W1.shape[1]
    abs_max = np.abs(W1).max()
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    n_cols = max(H, 3)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))
    fig.suptitle(
        "Part 3(a) — Templates vs Input->Hidden Weight Maps",
        fontsize=13, fontweight="bold"
    )

    # Row 0: character templates
    for j in range(n_cols):
        ax = axes[0][j]
        if j < 3:
            ax.imshow(templates[j], cmap="gray", vmin=-1, vmax=1,
                      interpolation="nearest")
            ax.set_title(f"Template  '{CLASS_NAMES[j]}'", fontsize=10)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

    # Row 1: weight heatmaps
    for h in range(n_cols):
        ax = axes[1][h]
        if h < H:
            w_img = W1[:, h].reshape(8, 8)
            im = ax.imshow(w_img, cmap="RdBu_r", norm=norm,
                           interpolation="nearest")
            ax.set_title(f"Hidden Unit {h + 1}\n"
                         f"(w range [{W1[:, h].min():.2f}, {W1[:, h].max():.2f}])",
                         fontsize=9, color=HIDDEN_COLORS[h % len(HIDDEN_COLORS)])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ─── (b) Hidden -> Output Weights ──────────────────────────────────────────────

def plot_hidden_output_weights(net: NeuralNetwork, save_path: str = None):
    """
    For each output unit (B, 0, E), display its H hidden-to-output weights
    as a grouped bar chart.  Also shows a heatmap matrix of the full W2.
    """
    W2 = net.get_weights()["W2"]   # shape (H, 3)
    H  = W2.shape[0]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        "Part 3(b) — Hidden -> Output Weights  (W2)",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    ax_bar = fig.add_subplot(gs[0])
    ax_hm  = fig.add_subplot(gs[1])

    # ── Grouped bar chart ─────────────────────────────────────────────────
    x       = np.arange(len(CLASS_NAMES))
    width   = 0.25
    offsets = np.linspace(-(H - 1) * width / 2, (H - 1) * width / 2, H)

    for h in range(H):
        bars = ax_bar.bar(x + offsets[h], W2[h, :], width,
                          label=f"Hidden Unit {h + 1}",
                          color=HIDDEN_COLORS[h % len(HIDDEN_COLORS)],
                          edgecolor="white", linewidth=0.6, alpha=0.88)
        for bar in bars:
            ht = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                ht + (0.005 if ht >= 0 else -0.02),
                f"{ht:.2f}", ha="center",
                va="bottom" if ht >= 0 else "top",
                fontsize=8, color="black"
            )

    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"Output '{n}'" for n in CLASS_NAMES], fontsize=11)
    ax_bar.set_ylabel("Weight Value", fontsize=10)
    ax_bar.set_title("Hidden->Output Weights per Output Unit", fontsize=11)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_ylim(W2.min() - 0.3, W2.max() + 0.3)

    # ── W2 heatmap matrix ─────────────────────────────────────────────────
    abs_max2 = np.abs(W2).max()
    norm2 = TwoSlopeNorm(vmin=-abs_max2, vcenter=0, vmax=abs_max2)
    im = ax_hm.imshow(W2, cmap="RdBu_r", norm=norm2, aspect="auto")

    ax_hm.set_xticks(range(3))
    ax_hm.set_yticks(range(H))
    ax_hm.set_xticklabels([f"'{n}'" for n in CLASS_NAMES], fontsize=11)
    ax_hm.set_yticklabels([f"H{h+1}" for h in range(H)], fontsize=10)
    ax_hm.set_xlabel("Output Unit (class)", fontsize=10)
    ax_hm.set_ylabel("Hidden Unit", fontsize=10)
    ax_hm.set_title("W2 Matrix\n(rows=hidden, cols=output)", fontsize=10)

    for i in range(H):
        for j in range(3):
            ax_hm.text(j, i, f"{W2[i, j]:.2f}",
                       ha="center", va="center", fontsize=10,
                       color="white" if abs(W2[i, j]) > abs_max2 * 0.5 else "black",
                       fontweight="bold")

    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.06)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ─── Bias visualisation ───────────────────────────────────────────────────────

def plot_biases(net: NeuralNetwork, save_path: str = None):
    """Display bias values for both layers as annotated bar charts."""
    w  = net.get_weights()
    b1 = w["b1"]
    b2 = w["b2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle("Bias Values — Both Layers", fontsize=12, fontweight="bold")

    ax1.bar(range(net.hidden_dim), b1,
            color=HIDDEN_COLORS[:net.hidden_dim],
            edgecolor="white", alpha=0.9)
    for i, v in enumerate(b1):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_xticks(range(net.hidden_dim))
    ax1.set_xticklabels([f"H{i+1}" for i in range(net.hidden_dim)])
    ax1.set_title("Hidden Layer Biases (b1)")
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.grid(axis="y", alpha=0.3)

    colors_out = ["#4C72B0", "#DD8452", "#55A868"]
    ax2.bar(range(3), b2, color=colors_out, edgecolor="white", alpha=0.9)
    for i, v in enumerate(b2):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([f"'{n}'" for n in CLASS_NAMES])
    ax2.set_title("Output Layer Biases (b2)")
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ─── Weight stats ─────────────────────────────────────────────────────────────

def print_weight_stats(net: NeuralNetwork):
    w = net.get_weights()
    W1, b1, W2, b2 = w["W1"], w["b1"], w["W2"], w["b2"]

    print("\n" + "=" * 55)
    print("  WEIGHT STATISTICS")
    print("=" * 55)
    for name, W in [("W1 (input->hidden)",  W1),
                    ("b1 (hidden bias)",    b1.reshape(1, -1)),
                    ("W2 (hidden->output)", W2),
                    ("b2 (output bias)",    b2.reshape(1, -1))]:
        print(f"  {name:25s}  "
              f"shape={str(W.shape):10s}  "
              f"min={W.min():+.3f}  max={W.max():+.3f}  "
              f"mean={W.mean():+.3f}  std={W.std():.3f}")
    print("=" * 55)

    print("\n  Per-hidden-unit weight summary (W1 columns):")
    for h in range(W1.shape[1]):
        col = W1[:, h]
        print(f"    Hidden {h+1}: min={col.min():+.3f}  max={col.max():+.3f}  "
              f"mean={col.mean():+.3f}  std={col.std():.3f}")

    print("\n  Per-output-unit weight summary (W2 columns):")
    for k, name in enumerate(CLASS_NAMES):
        col = W2[:, k]
        print(f"    Output '{name}': "
              + "  ".join([f"H{h+1}={col[h]:+.3f}" for h in range(len(col))]))


def print_interpretation(net: NeuralNetwork):
    w = net.get_weights()
    W1, W2 = w["W1"], w["W2"]
    H = W1.shape[1]

    print("\n" + "=" * 60)
    print("  WEIGHT INTERPRETATION")
    print("=" * 60)

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    net = NeuralNetwork.load("models/net_64_3_3")

    print_weight_stats(net)

    plot_input_hidden_weights(net,
        save_path="plots/weights_input_hidden.png")
    plot_input_hidden_overlay(net,
        save_path="plots/weights_overlay.png")
    plot_hidden_output_weights(net,
        save_path="plots/weights_hidden_output.png")
    plot_biases(net,
        save_path="plots/biases.png")

    print_interpretation(net)