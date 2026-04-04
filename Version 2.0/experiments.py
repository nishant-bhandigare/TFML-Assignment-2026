"""
(1.e) Search best 64–X–3 architecture; plots for accuracy vs X and sample complexity.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import generate_dataset
from train import TrainConfig, train_one_run


def run_architecture_sweep(
    X_full: np.ndarray,
    y_full: np.ndarray,
    hidden_sizes: list[int],
    epochs: int,
    device: torch.device,
    seed: int,
    n_runs: int,
) -> dict[int, list[float]]:
    """Multiple random seeds per X; return mean/std of final val accuracy."""
    results: dict[int, list[float]] = {}
    for h in hidden_sizes:
        finals: list[float] = []
        for r in range(n_runs):
            s = seed + r * 9973
            cfg = TrainConfig(
                hidden_dim=h,
                epochs=epochs,
                batch_size=32,
                lr=0.002,
                scheduler="onecycle",
            )
            *_, final_val_acc = train_one_run(X_full, y_full, device, s, cfg)
            finals.append(final_val_acc)
        results[h] = finals
    return results


def run_sample_complexity(
    hidden_dim: int,
    sample_counts: list[int],
    epochs: int,
    device: torch.device,
    base_seed: int,
    n_runs: int,
) -> tuple[list[float], list[float]]:
    """For each training set size, average final val accuracy (same 80/20 split logic on subset)."""
    means: list[float] = []
    stds: list[float] = []
    for n_train_total in sample_counts:
        # Approximately n_train_total/3 per class: round to multiples
        per = n_train_total // 3
        vals: list[float] = []
        for r in range(n_runs):
            s = base_seed + r * 7919
            X, y = generate_dataset(n_per_class=per, seed=s)
            cfg = TrainConfig(
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=min(32, len(X)),
                lr=0.002,
                scheduler="onecycle",
            )
            *_, final_val_acc = train_one_run(X, y, device, s + 1, cfg)
            vals.append(final_val_acc)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return means, stds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="outputs")
    ap.add_argument("--epochs-arch", type=int, default=300)
    ap.add_argument("--epochs-samples", type=int, default=350)
    ap.add_argument("--runs", type=int, default=5, help="Seeds per configuration for stability")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    X, y = generate_dataset(seed=args.seed)
    hidden_sizes = [2, 3, 4, 8, 16, 32, 64]
    arch_results = run_architecture_sweep(
        X, y, hidden_sizes, epochs=args.epochs_arch, device=device, seed=args.seed, n_runs=args.runs
    )

    means = [np.mean(arch_results[h]) for h in hidden_sizes]
    stds = [np.std(arch_results[h]) for h in hidden_sizes]
    best_h = hidden_sizes[int(np.argmax(means))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(hidden_sizes, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Hidden units X")
    ax.set_ylabel("Final validation accuracy")
    ax.set_title("64–X–3 MLP: validation accuracy vs hidden width (mean ± std over seeds)")
    ax.set_xticks(hidden_sizes)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(args.out_dir, "exp_architecture_vs_hidden.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    # Sample complexity: total samples 30..300 step 30
    sample_counts = list(range(30, 301, 30))
    sm, ss = run_sample_complexity(
        hidden_dim=best_h,
        sample_counts=sample_counts,
        epochs=args.epochs_samples,
        device=device,
        base_seed=args.seed,
        n_runs=args.runs,
    )
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.errorbar(sample_counts, sm, yerr=ss, marker="s", capsize=4, color="darkgreen")
    ax2.set_xlabel("Training set size (balanced: n/3 per class)")
    ax2.set_ylabel("Final validation accuracy")
    ax2.set_title(
        f"Sample complexity (hidden={best_h}): accuracy vs dataset size (mean ± std)"
    )
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(args.out_dir, "exp_sample_complexity.png")
    plt.savefig(p2, dpi=150)
    plt.close()

    report = os.path.join(args.out_dir, "experiment_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("Architecture sweep (validation accuracy, mean ± std):\n")
        for h in hidden_sizes:
            f.write(f"  X={h}: {np.mean(arch_results[h]):.4f} ± {np.std(arch_results[h]):.4f}\n")
        f.write(f"\nBest X by mean val acc: {best_h}\n")
        f.write("\nSample complexity (using best X):\n")
        for n, m, sd in zip(sample_counts, sm, ss):
            f.write(f"  N={n}: {m:.4f} ± {sd:.4f}\n")
    print(f"Plots saved: {p1}, {p2}")
    print(f"Report: {report}")
    print(f"Best hidden size (by mean val acc): {best_h}")


if __name__ == "__main__":
    main()
