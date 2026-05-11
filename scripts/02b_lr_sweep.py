#!/usr/bin/env python3
"""
Part 2 (supplemental): Learning Rate Sweep on the Tiny model.
Tests multiple learning rates and plots validation loss curves.

Usage:
    python scripts/02b_lr_sweep.py --data_dir data/processed --out_dir results/lr_sweep
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LR_CANDIDATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def run_lr_sweep(args):
    """Run the training for each LR candidate on the tiny model."""
    import subprocess, shlex

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for lr in LR_CANDIDATES:
        lr_str = f"{lr:.0e}".replace("+", "")
        run_dir = out_root / f"lr_{lr_str}"
        logger.info(f"\n{'='*50}\nLR = {lr}\n{'='*50}")

        cmd = (
            f"{sys.executable} {Path(__file__).parent}/02_train.py "
            f"--model tiny "
            f"--data_dir {args.data_dir} "
            f"--out_dir {run_dir} "
            f"--lr {lr}"
        )
        ret = subprocess.run(shlex.split(cmd))
        if ret.returncode != 0:
            logger.warning(f"Training failed for lr={lr}")
            continue

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            final_val = m.get("final_val_loss", float("inf"))
            results[lr_str] = {"lr": lr, "final_val_loss": final_val, "metrics": m}
            logger.info(f"LR={lr} → final val loss: {final_val:.4f}")

    # Save summary
    summary = {k: {"lr": v["lr"], "final_val_loss": v["final_val_loss"]} for k, v in results.items()}
    with open(out_root / "lr_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best_lr_str = min(summary, key=lambda k: summary[k]["final_val_loss"])
    best_lr = summary[best_lr_str]["lr"]
    logger.info(f"\nBest LR: {best_lr} (val loss = {summary[best_lr_str]['final_val_loss']:.4f})")

    # Plot
    plot_lr_sweep(results, out_root / "lr_sweep_plot.png")
    return best_lr, summary


def plot_lr_sweep(results: dict, output_path):
    """Plot val loss vs learning rate (log scale)."""
    lrs = [v["lr"] for v in results.values()]
    final_losses = [v["final_val_loss"] for v in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learning Rate Sweep — Tiny Model", fontsize=14, fontweight="bold")

    # Left: final val loss vs LR
    axes[0].plot(lrs, final_losses, "o-", color="#E91E63", linewidth=2, markersize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Final Validation Loss")
    axes[0].set_title("Val Loss vs Learning Rate")
    axes[0].grid(True, which="both", alpha=0.3)
    best_idx = int(np.argmin(final_losses))
    axes[0].axvline(lrs[best_idx], color="green", linestyle="--", label=f"Best LR={lrs[best_idx]:.0e}")
    axes[0].legend()

    # Right: training curves
    colors = plt.cm.plasma(np.linspace(0, 1, len(results)))
    for i, (lr_str, data) in enumerate(results.items()):
        m = data["metrics"]
        steps = m.get("steps", [])
        train_losses = m.get("train_losses", [])
        if steps and train_losses:
            axes[1].plot(steps, train_losses, color=colors[i], alpha=0.8,
                         label=f"lr={data['lr']:.0e}", linewidth=1.5)

    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Training Curves by LR")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved LR sweep plot: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning rate sweep")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="results/lr_sweep")
    args = parser.parse_args()

    best_lr, summary = run_lr_sweep(args)
    print(f"\nRecommended LR for scaling study: {best_lr}")
