#!/usr/bin/env python3
"""
Part 3 (supplemental): Compare SP vs µP scaling curves,
fit power laws, and make extrapolation predictions.

Usage:
    python scripts/03b_compare_scaling.py \
        --sp_dir checkpoints/sp \
        --mup_dir checkpoints/mup \
        --out_dir results
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "scaling_plot",
    os.path.join(os.path.dirname(__file__), "02c_scaling_plot.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
power_law = _mod.power_law
fit_scaling_law = _mod.fit_scaling_law
load_results = _mod.load_results
MODEL_SIZES_ORDER = _mod.MODEL_SIZES_ORDER
extrapolate = _mod.extrapolate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_and_plot(sp_results: dict, mup_results: dict, out_path: str):
    """Plot SP vs µP scaling curves on the same axes."""
    def extract(results):
        names = [k for k in MODEL_SIZES_ORDER if k in results]
        ns = np.array([results[k]["n_params"] for k in names], dtype=float)
        ls = np.array([results[k].get("final_val_loss", np.nan) for k in names])
        valid = ~np.isnan(ls)
        return ns[valid], ls[valid]

    sp_n, sp_l = extract(sp_results)
    mup_n, mup_l = extract(mup_results)

    sp_popt, sp_pcov = fit_scaling_law(sp_n, sp_l)
    mup_popt, mup_pcov = fit_scaling_law(mup_n, mup_l)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SP vs µP Scaling Comparison", fontsize=14, fontweight="bold")

    # Left: scaling curves
    ax = axes[0]
    n_range = np.logspace(
        np.log10(min(sp_n.min(), mup_n.min())),
        np.log10(max(sp_n.max(), mup_n.max())),
        300,
    )

    ax.scatter(sp_n, sp_l, s=120, color="#E91E63", zorder=5, label="SP (data)", marker="o")
    ax.scatter(mup_n, mup_l, s=120, color="#2196F3", zorder=5, label="µP (data)", marker="^")

    if sp_popt is not None:
        a, alpha, c = sp_popt
        ax.plot(n_range, power_law(n_range, a, alpha, c), "--", color="#E91E63",
                linewidth=2, label=f"SP fit: α={alpha:.3f}")

    if mup_popt is not None:
        a, alpha, c = mup_popt
        ax.plot(n_range, power_law(n_range, a, alpha, c), "--", color="#2196F3",
                linewidth=2, label=f"µP fit: α={alpha:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Validation Loss (1 epoch)", fontsize=12)
    ax.set_title("Scaling Law Comparison")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    # Right: difference (µP benefit over SP)
    ax2 = axes[1]
    if sp_popt is not None and mup_popt is not None:
        sp_curve = power_law(n_range, *sp_popt)
        mup_curve = power_law(n_range, *mup_popt)
        improvement = sp_curve - mup_curve
        ax2.plot(n_range, improvement, color="#4CAF50", linewidth=2)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax2.fill_between(n_range, 0, improvement, where=improvement > 0, alpha=0.3, color="#4CAF50", label="µP better")
        ax2.fill_between(n_range, 0, improvement, where=improvement < 0, alpha=0.3, color="#F44336", label="SP better")
        ax2.set_xscale("log")
        ax2.set_xlabel("Number of Parameters", fontsize=12)
        ax2.set_ylabel("SP Loss − µP Loss", fontsize=12)
        ax2.set_title("µP Improvement vs Model Size")
        ax2.legend(fontsize=10)
        ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")

    return sp_popt, sp_pcov, mup_popt, mup_pcov


def print_extrapolation(sp_popt, sp_pcov, mup_popt, mup_pcov, xl_n_params: float):
    """Extrapolate to 10× XL and report predictions."""
    target = xl_n_params * 10
    print(f"\n{'='*60}")
    print(f"Extrapolation to 10× XL model (N = {target:.2e})")
    print("-"*60)

    if sp_popt is not None:
        pred, unc = extrapolate(sp_popt, sp_pcov, target)
        print(f"  SP  → Predicted val loss = {pred:.4f} ± {unc:.4f}")

    if mup_popt is not None:
        pred, unc = extrapolate(mup_popt, mup_pcov, target)
        print(f"  µP  → Predicted val loss = {pred:.4f} ± {unc:.4f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_dir", default="checkpoints/sp")
    parser.add_argument("--mup_dir", default="checkpoints/mup")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sp_results = load_results(args.sp_dir)
    mup_results = load_results(args.mup_dir)

    sp_popt, sp_pcov, mup_popt, mup_pcov = compare_and_plot(
        sp_results, mup_results,
        str(Path(args.out_dir) / "scaling_comparison_sp_vs_mup.png"),
    )

    # Save fit summary
    summary = {}
    if sp_popt is not None:
        summary["sp"] = {"a": float(sp_popt[0]), "alpha": float(sp_popt[1]), "c": float(sp_popt[2])}
    if mup_popt is not None:
        summary["mup"] = {"a": float(mup_popt[0]), "alpha": float(mup_popt[1]), "c": float(mup_popt[2])}

    with open(Path(args.out_dir) / "scaling_fits.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Find XL n_params for extrapolation
    xl_n = sp_results.get("xl", mup_results.get("xl", {})).get("n_params", 88_000_000)
    print_extrapolation(sp_popt, sp_pcov, mup_popt, mup_pcov, float(xl_n))

    if sp_popt is not None and mup_popt is not None:
        print("\nScaling exponent comparison:")
        print(f"  SP  α = {sp_popt[1]:.4f}")
        print(f"  µP  α = {mup_popt[1]:.4f}")
        diff = mup_popt[1] - sp_popt[1]
        print(f"  Δα = {diff:+.4f} ({'µP steeper' if diff > 0 else 'SP steeper'})")
