"""Main training script for Version 3.0."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.generate_data import generate_dataset, save_dataset
from src.data.preprocess import standardize_train_val
from src.models.model import build_model
from src.training.evaluate import confusion_matrix_plot
from src.utils.helpers import set_global_seed, train_val_split_stratified


def train_main(seed: int = 42, hidden_units: int = 32):
    root = Path(__file__).resolve().parents[2]
    set_global_seed(seed)

    X, y_idx, y_onehot = generate_dataset(samples_per_class=100, seed=seed)
    save_dataset(X, y_idx, y_onehot, root / "data" / "processed")

    X_train, y_train, X_val, y_val = train_val_split_stratified(X, y_idx, val_frac=0.2, seed=seed)
    X_train, X_val, mean, std = standardize_train_val(X_train, X_val)

    np.save(root / "data" / "processed" / "norm_stats.npy", np.array([mean, std], dtype=np.float32))

    model = build_model(hidden_units=hidden_units, dropout_rate=0.05)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(root / "models" / "checkpoints" / "best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(root / "models" / "trained_model.keras")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc * 100:.2f}%  | loss: {val_loss:.4f}")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    out_plot = root / "reports" / "figures" / "training_curves_v3.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    confusion_matrix_plot(model, X_val, y_val, root / "reports" / "figures" / "confusion_matrix_v3.png")


if __name__ == "__main__":
    train_main(seed=42, hidden_units=32)
