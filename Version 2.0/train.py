"""
Train 64–hidden–3 MLP on dataset D; save checkpoint and weight visualizations (assignment 1.a–1.c).

Improvements for accuracy: AdamW, label smoothing, cosine or OneCycle LR, gradient clipping,
optional LayerNorm + GELU + dropout, restore weights from best validation epoch.
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from dataset import generate_dataset
from model import LetterMLP
from visualize_weights import plot_hidden_output_weights, plot_input_hidden_weights


@dataclass
class TrainConfig:
    hidden_dim: int = 32
    epochs: int = 2000
    batch_size: int = 32
    lr: float = 0.002
    weight_decay: float = 1e-4
    label_smoothing: float = 0.08
    dropout: float | None = None  # None = auto from hidden_dim
    use_layer_norm: bool = True
    activation: str = "gelu"
    grad_clip_norm: float = 1.0
    scheduler: str = "onecycle"  # onecycle | cosine | none
    use_best_checkpoint: bool = True
    standardize: bool = True


def auto_dropout(hidden_dim: int) -> float:
    """Small nets need little or no dropout; wider nets benefit from mild dropout."""
    if hidden_dim <= 4:
        return 0.0
    if hidden_dim <= 8:
        return 0.05
    if hidden_dim <= 24:
        return 0.1
    return 0.15


def train_one_run(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    seed: int,
    config: TrainConfig,
) -> tuple[LetterMLP, list[float], list[float], np.ndarray | None, np.ndarray | None, float]:
    """Train with stratified split; optional input standardization on train only; return best-val model if enabled."""
    rng = np.random.default_rng(seed)
    tr_parts: list[np.ndarray] = []
    va_parts: list[np.ndarray] = []
    for c in range(int(y.max()) + 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        tr_parts.append(idx[:split])
        va_parts.append(idx[split:])
    tr = np.concatenate(tr_parts)
    va = np.concatenate(va_parts)
    X_tr_np, X_va_np = X[tr], X[va]
    mu: np.ndarray | None = None
    std: np.ndarray | None = None
    if config.standardize:
        mu = X_tr_np.mean(axis=0, keepdims=True)
        std = X_tr_np.std(axis=0, keepdims=True) + 1e-8
        X_tr_np = (X_tr_np - mu) / std
        X_va_np = (X_va_np - mu) / std
    X_tr, y_tr = torch.tensor(X_tr_np, dtype=torch.float32), torch.tensor(y[tr], dtype=torch.long)
    X_va, y_va = torch.tensor(X_va_np, dtype=torch.float32), torch.tensor(y[va], dtype=torch.long)

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    drop = config.dropout if config.dropout is not None else auto_dropout(config.hidden_dim)
    model = LetterMLP(
        hidden_dim=config.hidden_dim,
        dropout=drop,
        use_layer_norm=config.use_layer_norm,
        activation=config.activation,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    scheduler = None
    steps_per_epoch = len(loader)
    if config.scheduler == "onecycle":
        scheduler = OneCycleLR(
            opt,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.12,
            div_factor=25.0,
            final_div_factor=1e4,
        )
    elif config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=config.epochs)

    train_losses: list[float] = []
    val_accs: list[float] = []
    best_val = -1.0
    best_state: dict | None = None

    torch.manual_seed(seed)

    for ep in range(config.epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            opt.step()
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(ds)
        train_losses.append(ep_loss)

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_va.to(device))
            pred = logits.argmax(dim=1)
            acc = (pred.cpu() == y_va).float().mean().item()
        val_accs.append(acc)

        if acc > best_val:
            best_val = acc
            if config.use_best_checkpoint:
                best_state = copy.deepcopy(model.state_dict())

    if config.use_best_checkpoint and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_va.to(device))
        pred = logits.argmax(dim=1)
        final_val_acc = (pred.cpu() == y_va).float().mean().item()

    return model, train_losses, val_accs, mu, std, final_val_acc


def main() -> None:
    p = argparse.ArgumentParser(description="Train Letter MLP (TFML Assignment 2)")
    p.add_argument("--hidden", type=int, default=32, help="Hidden width X (use 3 for strict 1.a)")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.08)
    p.add_argument("--dropout", type=float, default=-1.0, help=">=0 fixed; -1 = auto by hidden size")
    p.add_argument("--no-layer-norm", action="store_true")
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu", "tanh", "silu"])
    p.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "cosine", "none"])
    p.add_argument("--no-best-checkpoint", action="store_true", help="Keep last epoch weights instead of best val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs")
    args = p.parse_args()

    drop = None if args.dropout < 0 else args.dropout
    cfg = TrainConfig(
        hidden_dim=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=drop,
        use_layer_norm=not args.no_layer_norm,
        activation=args.activation,
        scheduler=args.scheduler,
        use_best_checkpoint=not args.no_best_checkpoint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = generate_dataset(seed=args.seed)

    model, train_losses, val_accs, mu, std, final_val_acc = train_one_run(X, y, device, args.seed, cfg)
    best_val_during = max(val_accs)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"model_h{args.hidden}.pt")
    effective_drop = cfg.dropout if cfg.dropout is not None else auto_dropout(args.hidden)
    torch.save(
        {
            "version": 2,
            "model_state": model.state_dict(),
            "hidden_dim": args.hidden,
            "dropout": effective_drop,
            "use_layer_norm": cfg.use_layer_norm,
            "activation": cfg.activation,
            "train_losses": train_losses,
            "val_accs": val_accs,
            "final_val_acc": final_val_acc,
            "best_val_acc": best_val_during,
            "mu": mu,
            "std": std,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Validation acc of saved model: {final_val_acc:.4f} (best epoch in-run: {best_val_during:.4f})")
    print(f"Final train loss: {train_losses[-1]:.4f}")

    prefix = f"h{args.hidden}"
    plot_input_hidden_weights(model, args.out_dir, prefix=prefix)
    plot_hidden_output_weights(model, args.out_dir, prefix=prefix)
    print(f"Weight figures saved under {args.out_dir}/")


if __name__ == "__main__":
    main()
