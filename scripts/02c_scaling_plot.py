#!/usr/bin/env python3
"""
Part 2 (supplemental): Scaling plot — fit power law L = a * N^(-alpha) + c
to validation losses from all trained model sizes.

Usage:
    python scripts/02c_scaling_plot.py --checkpoints_dir checkpoints/sp --out_dir results
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SIZES_ORDER = ["tiny", "small", "medium", "large", "xl"]


def power_law(N, a, alpha, c):
    """L = a * N^(-alpha) + c"""
    return a * N**(-alpha) + c


def fit_scaling_law(n_params: np.ndarray, val_losses: np.ndarray):
    """Fit L = a * N^(-alpha) + c and return params + covariance."""
    try:
        p0 = [1.0, 0.3, 1.5]
        popt, pcov = curve_fit(power_law, n_params, val_losses, p0=p0, maxfev=10000)
        return popt, pcov
    except Exception as e:
        logger.warning(f"Curve fit failed: {e}")
        return None, None


def load_results(checkpoints_dir: str) -> dict:
    """Load metrics.json from each model size subdirectory."""
    ckpt_dir = Path(checkpoints_dir)
    results = {}
    for model_name in MODEL_SIZES_ORDER:
        metrics_path = ckpt_dir / model_name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            results[model_name] = m
            logger.info(f"  Loaded: {model_name} | n_params={m['n_params']:,} | val_loss={m.get('final_val_loss', '?')}")
        else:
            logger.warning(f"  Missing: {metrics_path}")
    return results


def plot_scaling(results: dict, out_path: str, title: str = "Standard Parameterization Scaling"):
    """Create scaling plot with power-law fit."""
    names = [k for k in MODEL_SIZES_ORDER if k in results]
    n_params = np.array([results[k]["n_params"] for k in names], dtype=float)
    val_losses = np.array([results[k].get("final_val_loss", results[k].get("best_val_loss", np.nan)) for k in names])

    # Remove NaNs
    valid = ~np.isnan(val_losses)
    n_params = n_params[valid]
    val_losses = val_losses[valid]

    popt, pcov = fit_scaling_law(n_params, val_losses)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = ["#E91E63", "#9C27B0", "#3F51B5", "#00BCD4", "#4CAF50"]

    # Left: log-log scaling plot
    ax = axes[0]
    for i, (nm, n, l) in enumerate(zip(names, n_params, val_losses)):
        ax.scatter(n, l, s=120, color=colors[i % len(colors)], zorder=5, label=nm)
        ax.annotate(nm, (n, l), textcoords="offset points", xytext=(5, 5), fontsize=9)

    if popt is not None:
        a, alpha, c = popt
        n_range = np.logspace(np.log10(n_params.min()), np.log10(n_params.max()), 200)
        l_range = power_law(n_range, a, alpha, c)
        ax.plot(n_range, l_range, "--", color="gray", linewidth=2,
                label=f"Fit: L = {a:.2f}·N⁻{alpha:.3f} + {c:.2f}")

        # Confidence interval via std of params
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
            l_upper = power_law(n_range, a + perr[0], alpha - perr[1], c + perr[2])
            l_lower = power_law(n_range, a - perr[0], alpha + perr[1], c - perr[2])
            ax.fill_between(n_range, l_lower, l_upper, alpha=0.15, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (non-embedding)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Scaling Law: Val Loss vs N Params")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Right: training curves
    ax2 = axes[1]
    for i, nm in enumerate(names):
        m = results[nm]
        steps = m.get("steps", [])
        train_losses = m.get("train_losses", [])
        if steps and train_losses:
            ax2.plot(steps, train_losses, color=colors[i % len(colors)],
                     label=nm, linewidth=1.8, alpha=0.85)

    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Training Loss", fontsize=12)
    ax2.set_title("Training Loss Curves")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")

    if popt is not None:
        a, alpha, c = popt
        logger.info(f"\nPower-law fit: L = {a:.4f} × N^(-{alpha:.4f}) + {c:.4f}")
        logger.info(f"  Scaling exponent α = {alpha:.4f}")

    return popt, pcov


def extrapolate(popt, pcov, target_n: float) -> tuple[float, float]:
    """Predict loss for a model with target_n parameters."""
    if popt is None:
        return float("nan"), float("nan")
    a, alpha, c = popt
    predicted = power_law(target_n, a, alpha, c)

    # Uncertainty via error propagation
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(3)
    da = perr[0] * target_n**(-alpha)
    dalpha = perr[1] * a * target_n**(-alpha) * abs(np.log(target_n))
    dc = perr[2]
    uncertainty = np.sqrt(da**2 + dalpha**2 + dc**2)

    return predicted, uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="checkpoints/sp")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--extra_n_params", type=float, default=None,
                        help="Extrapolate to this many params (e.g. 880000000 for 10x XL)")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    results = load_results(args.checkpoints_dir)
    if not results:
        logger.error("No results found. Run training first.")
        sys.exit(1)

    popt, pcov = plot_scaling(results, str(Path(args.out_dir) / "scaling_plot_sp.png"))

    # Print model table
    print("\n" + "="*70)
    print(f"{'Model':<10} {'Params':>12} {'Val Loss':>10} {'Time (s)':>10} {'GPU (GB)':>10}")
    print("-"*70)
    for nm in MODEL_SIZES_ORDER:
        if nm not in results:
            continue
        m = results[nm]
        vl = m.get("final_val_loss", float("nan"))
        t = m.get("wall_clock_seconds", float("nan"))
        gpu = max(m.get("gpu_memory_gb", [0.0]) or [0.0])
        print(f"{nm:<10} {m['n_params']:>12,} {vl:>10.4f} {t:>10.1f} {gpu:>10.2f}")
    print("="*70)

    if args.extra_n_params and popt is not None:
        pred, unc = extrapolate(popt, pcov, args.extra_n_params)
        print(f"\nExtrapolation to N={args.extra_n_params:.2e}:")
        print(f"  Predicted val loss = {pred:.4f} ± {unc:.4f}")

    # Save fit params
    if popt is not None:
        fit_data = {
            "a": float(popt[0]),
            "alpha": float(popt[1]),
            "c": float(popt[2]),
            "pcov": pcov.tolist() if pcov is not None else None,
        }
        import json
        with open(Path(args.out_dir) / "scaling_fit_sp.json", "w") as f:
            json.dump(fit_data, f, indent=2)
