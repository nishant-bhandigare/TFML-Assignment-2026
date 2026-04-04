"""
train.py
========
Part 2: Training loop, evaluation utilities, and loss/accuracy curve plots.

Trains the 64-3-3 network using Adam optimiser with mini-batch gradient descent.
Records per-epoch loss and accuracy; saves the best model by validation loss.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def batch_iter(X, y, batch_size: int, rng):
    """Yield (X_batch, y_batch) mini-batches with random shuffle each epoch."""
    N = len(X)
    perm = rng.permutation(N)
    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(net: NeuralNetwork, X, y_onehot, labels):
    """Return (loss, accuracy) on a dataset."""
    probs = net.predict_proba(X)
    loss  = net.cross_entropy_loss(probs, y_onehot)
    preds = np.argmax(probs, axis=1)
    acc   = np.mean(preds == labels)
    return loss, acc


# ─── Training Loop ────────────────────────────────────────────────────────────

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
    Train the network with Adam + mini-batch SGD.

    Early stopping: halt if validation loss has not improved for `patience` epochs.

    Returns
    -------
    history : dict with keys
        train_loss, train_acc, val_loss, val_acc  (lists, one entry per epoch)
        best_epoch  (int)
    """
    rng = np.random.default_rng(seed)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_loss  = np.inf
    best_weights   = None
    patience_count = 0

    for epoch in range(1, epochs + 1):

        # ── Mini-batch forward + backward ────────────────────────────────
        for X_b, y_b in batch_iter(X_train, y_train, batch_size, rng):
            cache = net.forward(X_b)
            grads = net.backward(cache, y_b)
            net._adam_update(grads, lr=lr, beta1=beta1, beta2=beta2)

        # ── Full-dataset metrics ──────────────────────────────────────────
        t_loss, t_acc = evaluate(net, X_train, y_train, labels_train)
        v_loss, v_acc = evaluate(net, X_val,   y_val,   labels_val)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        # ── Early stopping ────────────────────────────────────────────────
        if v_loss < best_val_loss - 1e-6:
            best_val_loss  = v_loss
            best_weights   = {k: v.copy() for k, v in net.get_weights().items()}
            patience_count = 0
            history["best_epoch"] = epoch
        else:
            patience_count += 1

        if patience_count >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}  "
                      f"(best val loss {best_val_loss:.4f} @ epoch {history['best_epoch']})")
            break

        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4d} | "
                  f"train loss {t_loss:.4f}  acc {t_acc*100:.1f}% | "
                  f"val loss {v_loss:.4f}  acc {v_acc*100:.1f}%")

    # Restore best weights
    if best_weights is not None:
        net.set_weights(best_weights)

    return history


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, title: str = "64-3-3 Network",
                         save_path: str = None):
    """
    Plot loss and accuracy curves (train + validation) side by side.
    Marks the best epoch with a vertical dashed line.
    """
    epochs_run  = len(history["train_loss"])
    best_epoch  = history.get("best_epoch", epochs_run)
    epoch_range = np.arange(1, epochs_run + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Loss curve ────────────────────────────────────────────────────────
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

    # ── Accuracy curve ────────────────────────────────────────────────────
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
        print(f"Training curves saved -> {save_path}")
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
        print(f"Confusion matrix saved -> {save_path}")
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

    # ── Load data ─────────────────────────────────────────────────────────
    print("[1] Loading dataset …")
    X, y, labels = load_dataset("data")
    print(f"    X: {X.shape}  y: {y.shape}  labels: {labels.shape}")

    # ── Train / val split (80/20 stratified) ─────────────────────────────
    X_tr, y_tr, l_tr, X_v, y_v, l_v = train_val_split(X, y, labels,
                                                        val_frac=0.2, seed=7)
    print(f"    Train: {len(X_tr)}  Val: {len(X_v)}")

    # ── Build 64-3-3 network ──────────────────────────────────────────────
    print("\n[2] Building 64-3-3 network …")
    net = NeuralNetwork(input_dim=64, hidden_dim=3, output_dim=3, seed=42)
    print(f"    {net}")
    print(f"    W1 shape: {net.W1.shape}   W2 shape: {net.W2.shape}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[3] Training (Adam, lr=0.001, max 3000 epochs, patience=300) …\n")
    history = train(
        net,
        X_tr, y_tr, l_tr,
        X_v,  y_v,  l_v,
        epochs=3000, batch_size=32,
        lr=0.001, beta1=0.9, beta2=0.999,
        patience=300, seed=0, verbose=True,
    )

    # ── Final metrics ──────────────────────────────────────────────────────
    print_final_metrics(net, history, X_tr, y_tr, l_tr, X_v, y_v, l_v)

    # ── Save model ─────────────────────────────────────────────────────────
    print("\n[4] Saving model …")
    net.save("models/net_64_3_3")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[5] Plotting training curves …")
    plot_training_curves(history,
                         title="64-3-3 Network — Training & Validation Curves",
                         save_path="plots/training_curves.png")

    print("\n[6] Plotting confusion matrix …")
    X_all = np.vstack([X_tr, X_v])
    l_all = np.concatenate([l_tr, l_v])
    plot_confusion_matrix(net, X_all, l_all,
                          title="Confusion Matrix (Full Dataset)",
                          save_path="plots/confusion_matrix.png")

    print("\nPart 2 complete.")
