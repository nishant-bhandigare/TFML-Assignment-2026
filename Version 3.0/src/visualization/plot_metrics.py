"""Utility plotting helpers for metrics arrays."""

import matplotlib.pyplot as plt


def save_line_plot(x, y, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
