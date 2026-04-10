"""Architecture and sample complexity experiments for part (e)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.generate_data import generate_dataset
from src.data.preprocess import standardize_train_val
from src.models.model import build_model
from src.utils.helpers import train_val_split_stratified, set_global_seed


def run_architecture_search(seed: int = 42):
    root = Path(__file__).resolve().parents[2]
    set_global_seed(seed)
    hidden_sizes = [4, 8, 16, 32, 64, 128]
    X, y_idx, _ = generate_dataset(samples_per_class=100, seed=seed)
    X_train, y_train, X_val, y_val = train_val_split_stratified(X, y_idx, val_frac=0.2, seed=seed)
    X_train, X_val, _, _ = standardize_train_val(X_train, X_val)

    val_scores = []
    for h in hidden_sizes:
        model = build_model(hidden_units=h, dropout_rate=0.1)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=120, batch_size=16, verbose=0)
        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        val_scores.append(val_acc)
        print(f"Hidden {h:>3}: val_acc={val_acc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hidden_sizes, [v * 100 for v in val_scores], marker="o")
    ax.set_xlabel("Hidden units (X)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Architecture Search: 64-X-3")
    fig.tight_layout()
    fig.savefig(root / "reports" / "figures" / "architecture_search_v3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sample_complexity(seed: int = 42, hidden_units: int = 32):
    root = Path(__file__).resolve().parents[2]
    set_global_seed(seed)
    per_class_sizes = [10, 20, 50, 100, 200]
    val_scores = []

    for n in per_class_sizes:
        X, y_idx, _ = generate_dataset(samples_per_class=n, seed=seed)
        X_train, y_train, X_val, y_val = train_val_split_stratified(X, y_idx, val_frac=0.2, seed=seed)
        X_train, X_val, _, _ = standardize_train_val(X_train, X_val)
        model = build_model(hidden_units=hidden_units, dropout_rate=0.1)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=120, batch_size=16, verbose=0)
        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        val_scores.append(val_acc)
        print(f"Samples/class {n:>3}: val_acc={val_acc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(per_class_sizes, [v * 100 for v in val_scores], marker="o")
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Sample Complexity (X={hidden_units})")
    fig.tight_layout()
    fig.savefig(root / "reports" / "figures" / "sample_complexity_v3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_architecture_search()
    run_sample_complexity(hidden_units=32)
