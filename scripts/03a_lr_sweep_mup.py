#!/usr/bin/env python3
"""
Part 3 (supplemental): µP Learning Rate Sweep on the Tiny model.
Same protocol as 02b_lr_sweep.py but uses µP parameterization.
Generates a side-by-side comparison plot of SP vs µP LR sweeps.

Usage:
    python scripts/03a_lr_sweep_mup.py \
        --data_dir data/processed \
        --sp_sweep_dir results/lr_sweep \
        --out_dir results/lr_sweep_mup
"""

import sys
import json
import argparse
import logging
import subprocess
import shlex
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LR_CANDIDATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def run_mup_sweep(args):
    """Train tiny µP model at each LR candidate and collect val losses."""
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for lr in LR_CANDIDATES:
        lr_str = f"{lr:.0e}".replace("+", "")
        run_dir = out_root / f"lr_{lr_str}"
        logger.info(f"\n{'='*50}\n[µP] LR = {lr}\n{'='*50}")

        cmd = (
            f"{sys.executable} {Path(__file__).parent}/03_train_mup.py "
            f"--model tiny "
            f"--data_dir {args.data_dir} "
            f"--out_dir {run_dir} "
            f"--lr {lr}"
        )
        ret = subprocess.run(shlex.split(cmd))
        if ret.returncode != 0:
            logger.warning(f"[µP] Training failed for lr={lr}")
            continue

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            final_val = m.get("final_val_loss", float("inf"))
            results[lr_str] = {"lr": lr, "final_val_loss": final_val, "metrics": m}
            logger.info(f"[µP] LR={lr} → final val loss: {final_val:.4f}")

    summary = {k: {"lr": v["lr"], "final_val_loss": v["final_val_loss"]} for k, v in results.items()}
    with open(out_root / "mup_lr_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best_lr_str = min(summary, key=lambda k: summary[k]["final_val_loss"])
    best_lr = summary[best_lr_str]["lr"]
    logger.info(f"\n[µP] Best LR: {best_lr} (val loss = {summary[best_lr_str]['final_val_loss']:.4f})")

    return best_lr, results, summary


def plot_sp_vs_mup_lr_sweep(sp_sweep_dir: str, mup_results: dict, out_path: str):
    """
    Plot SP vs µP final val loss vs LR on same axes.
    Shows whether µP achieves a flatter / more stable LR sensitivity curve.
    """
    # Load SP sweep summary
    sp_summary_path = Path(sp_sweep_dir) / "lr_sweep_summary.json"
    if not sp_summary_path.exists():
        logger.warning(f"SP sweep summary not found at {sp_summary_path}")
        sp_summary = {}
    else:
        with open(sp_summary_path) as f:
            sp_summary = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("LR Sweep Comparison: Standard Parameterization vs µP (Tiny Model)", fontsize=13, fontweight="bold")

    # ── Left: val loss vs LR ──────────────────────────────────────
    ax = axes[0]

    if sp_summary:
        sp_lrs = [v["lr"] for v in sp_summary.values()]
        sp_losses = [v["final_val_loss"] for v in sp_summary.values()]
        ax.plot(sp_lrs, sp_losses, "o-", color="#E91E63", linewidth=2, markersize=9, label="SP")
        best_sp_idx = int(np.argmin(sp_losses))
        ax.axvline(sp_lrs[best_sp_idx], color="#E91E63", linestyle=":", alpha=0.6,
                   label=f"SP best: {sp_lrs[best_sp_idx]:.0e}")

    if mup_results:
        mup_lrs = [v["lr"] for v in mup_results.values()]
        mup_losses = [v["final_val_loss"] for v in mup_results.values()]
        ax.plot(mup_lrs, mup_losses, "^-", color="#2196F3", linewidth=2, markersize=9, label="µP")
        best_mup_idx = int(np.argmin(mup_losses))
        ax.axvline(mup_lrs[best_mup_idx], color="#2196F3", linestyle=":", alpha=0.6,
                   label=f"µP best: {mup_lrs[best_mup_idx]:.0e}")

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Final Validation Loss (1 epoch)", fontsize=12)
    ax.set_title("Val Loss vs LR — Tiny Model")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    # ── Right: training curves at best LR for each ────────────────
    ax2 = axes[1]

    if sp_summary:
        best_sp_key = min(sp_summary, key=lambda k: sp_summary[k]["final_val_loss"])
        sp_metrics_path = Path(sp_sweep_dir) / f"lr_{best_sp_key}" / "metrics.json"
        if sp_metrics_path.exists():
            with open(sp_metrics_path) as f:
                sp_m = json.load(f)
            ax2.plot(sp_m.get("steps", []), sp_m.get("train_losses", []),
                     color="#E91E63", linewidth=2, label=f"SP (lr={sp_summary[best_sp_key]['lr']:.0e})")

    if mup_results:
        best_mup_key = min(mup_results, key=lambda k: mup_results[k]["final_val_loss"])
        mup_m = mup_results[best_mup_key]["metrics"]
        ax2.plot(mup_m.get("steps", []), mup_m.get("train_losses", []),
                 color="#2196F3", linewidth=2, label=f"µP (lr={mup_results[best_mup_key]['lr']:.0e})")

    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Training Loss", fontsize=12)
    ax2.set_title("Training Curves at Best LR")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="µP LR sweep on Tiny model")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--sp_sweep_dir", default="results/lr_sweep",
                        help="Directory of the SP LR sweep (for comparison plot)")
    parser.add_argument("--out_dir", default="results/lr_sweep_mup")
    args = parser.parse_args()

    best_lr, mup_results, mup_summary = run_mup_sweep(args)

    # Comparison plot
    plot_sp_vs_mup_lr_sweep(
        args.sp_sweep_dir,
        mup_results,
        Path(args.out_dir) / "lr_sweep_sp_vs_mup.png",
    )

    print(f"\n[µP] Recommended LR for µP scaling study: {best_lr}")
