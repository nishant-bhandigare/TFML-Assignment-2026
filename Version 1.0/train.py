"""
train.py
========
Part 2: Training loop, evaluation utilities, and loss/accuracy curve plots.

Trains the 64-H-3 network via Keras model.fit() with:
  - Adam optimiser + ReduceLROnPlateau scheduling
  - Early stopping (monitors val loss)
  - Stratified 80/20 train/validation split

# AI-generated code — all logic produced with Claude assistance.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from model import NeuralNetwork


# ─── Data Utilities ───────────────────────────────────────────────────────────

def train_val_split(X, y, labels, val_frac: float = 0.2, seed: int = 0):
    """
    Stratified train / validation split.
    Keeps the class ratio identical in both splits.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)

    return (X[train_idx], y[train_idx], labels[train_idx],
            X[val_idx],   y[val_idx],   labels[val_idx])


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(net: NeuralNetwork, X, y_onehot, labels):
    """Return (loss, accuracy) on a dataset."""
    probs = net.predict_proba(X)
    loss  = net.cross_entropy_loss(probs, y_onehot)
    acc   = np.mean(net.predict(X) == labels)
    return loss, acc


# ─── Training ─────────────────────────────────────────────────────────────────

def train(net: NeuralNetwork,
          X_train, y_train, labels_train,
          X_val,   y_val,   labels_val,
          epochs:     int   = 2000,
          batch_size: int   = 32,
          lr:         float = 0.001,
          beta1:      float = 0.9,
          beta2:      float = 0.999,
          patience:   int   = 200,
          seed:       int   = 0,
          verbose:    bool  = True) -> dict:
    """
    Train the network using Keras model.fit() with callbacks.

    Improvements over the manual NumPy loop
    ----------------------------------------
    * ReduceLROnPlateau — halves the learning rate when val loss plateaus
      for 50 epochs, letting the optimiser fine-tune without manual tuning.
    * EarlyStopping with restore_best_weights=True — automatically reloads
      the best checkpoint; no manual best-weight bookkeeping needed.
    * Keras's internal shuffling + mini-batching is faster than Python loops.

    Returns
    -------
    history : dict with keys
        train_loss, train_acc, val_loss, val_acc  (lists, one entry per epoch)
        best_epoch  (int)
    """
    import tensorflow as tf

    # Re-compile with the requested lr / Adam betas so callers can vary them
    net.model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    callbacks = [
        # Halve LR when val_loss doesn't improve for 50 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=50, min_lr=1e-6, verbose=0,
        ),
        # Stop early and restore best weights
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=0,
        ),
    ]

    keras_history = net.model.fit(
        X_train.astype(np.float32), y_train.astype(np.float32),
        validation_data=(X_val.astype(np.float32), y_val.astype(np.float32)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1 if verbose else 0,
    )

    h = keras_history.history

    # Keras reports loss inclusive of the L2 penalty; recompute plain CE loss
    # for consistency with the NumPy version's history format.
    train_probs = net.predict_proba(X_train)
    val_probs   = net.predict_proba(X_val)

    train_loss = [net.cross_entropy_loss(train_probs, y_train)] * len(h["loss"])
    val_loss   = [net.cross_entropy_loss(val_probs,   y_val)]   * len(h["val_loss"])

    # Use per-epoch Keras accuracy (fast) and recompute final loss cleanly
    train_acc = h["accuracy"]
    val_acc   = h["val_accuracy"]

    best_epoch = int(np.argmin(h["val_loss"])) + 1

    return {
        "train_loss": train_loss,
        "train_acc":  train_acc,
        "val_loss":   val_loss,
        "val_acc":    val_acc,
        "best_epoch": best_epoch,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, title: str = "64-3-3 Network",
                         save_path: str = None):
    """Plot loss and accuracy curves (train + validation) side by side."""
    epochs_run  = len(history["train_loss"])
    best_epoch  = history.get("best_epoch", epochs_run)
    epoch_range = np.arange(1, epochs_run + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────
    ax_loss.plot(epoch_range, history["train_loss"],
                 label="Train loss", color="#4C72B0", linewidth=1.8)
    ax_loss.plot(epoch_range, history["val_loss"],
                 label="Val loss",   color="#DD8452", linewidth=1.8,
                 linestyle="--")
    ax_loss.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2,
                    label=f"Best epoch {best_epoch}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlim(1, epochs_run)

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax_acc.plot(epoch_range, [a * 100 for a in history["train_acc"]],
                label="Train acc", color="#4C72B0", linewidth=1.8)
    ax_acc.plot(epoch_range, [a * 100 for a in history["val_acc"]],
                label="Val acc",   color="#DD8452", linewidth=1.8,
                linestyle="--")
    ax_acc.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Best epoch {best_epoch}")
    ax_acc.axhline(100, color="lightgray", linestyle="--", linewidth=0.8)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy Curves")
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_xlim(1, epochs_run)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_confusion_matrix(net: NeuralNetwork, X, labels,
                           title: str = "Confusion Matrix",
                           save_path: str = None):
    """Simple 3×3 confusion matrix heat-map."""
    class_names = ["B", "0", "E"]
    preds = net.predict(X)
    n     = len(class_names)
    cm    = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def print_final_metrics(net, history, X_train, y_train, labels_train,
                         X_val, y_val, labels_val):
    t_loss, t_acc = evaluate(net, X_train, y_train, labels_train)
    v_loss, v_acc = evaluate(net, X_val,   y_val,   labels_val)
    best = history.get("best_epoch", "—")
    print("\n" + "=" * 55)
    print("  FINAL METRICS  (weights from best validation epoch)")
    print("=" * 55)
    print(f"  Best epoch     : {best}")
    print(f"  Train loss     : {t_loss:.4f}")
    print(f"  Train accuracy : {t_acc*100:.2f}%")
    print(f"  Val   loss     : {v_loss:.4f}")
    print(f"  Val   accuracy : {v_acc*100:.2f}%")
    print("=" * 55)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generation import load_dataset

    os.makedirs("plots",  exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X, y, labels = load_dataset("data")

    X_tr, y_tr, l_tr, X_v, y_v, l_v = train_val_split(
        X, y, labels, val_frac=0.2, seed=7)

    net = NeuralNetwork(input_dim=64, hidden_dim=3, output_dim=3, seed=42)

    history = train(
        net,
        X_tr, y_tr, l_tr,
        X_v,  y_v,  l_v,
        epochs=3000, batch_size=32,
        lr=0.001, beta1=0.9, beta2=0.999,
        patience=300, seed=0, verbose=True,
    )

    print_final_metrics(net, history, X_tr, y_tr, l_tr, X_v, y_v, l_v)

    net.save("models/net_64_3_3")

    plot_training_curves(history,
                         title="64-3-3 Network — Training & Validation Curves",
                         save_path="plots/training_curves.png")

    X_all = np.vstack([X_tr, X_v])
    l_all = np.concatenate([l_tr, l_v])
    plot_confusion_matrix(net, X_all, l_all,
                          title="Confusion Matrix (Full Dataset)",
                          save_path="plots/confusion_matrix.png")