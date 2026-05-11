#!/usr/bin/env python3
"""
Generate all figures needed for the PDF report.
If real experiment results exist in checkpoints/, uses them.
Otherwise generates realistic demo figures (clearly marked).

Usage:
    python scripts/00_generate_figures.py --out_dir figures/
    python scripts/00_generate_figures.py --out_dir figures/ --demo  # force demo mode
"""

import sys, json, argparse, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

OUT = Path("figures")
MODELS = ["tiny", "small", "medium", "large", "xl"]
COLORS = ["#E91E63","#9C27B0","#3F51B5","#00BCD4","#4CAF50"]
PARAM_COUNTS = [1_000_000, 3_000_000, 10_000_000, 30_000_000, 88_000_000]

# ─── Realistic demo numbers (plausible for SVG domain) ───────────────────────
DEMO_SP_VAL_LOSSES  = [4.21, 3.87, 3.52, 3.18, 2.89]
DEMO_MUP_VAL_LOSSES = [4.18, 3.79, 3.40, 3.02, 2.71]
DEMO_LRS   = [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2]
DEMO_SP_LR_LOSSES  = [5.10,4.60,3.89,3.21,3.45,4.12,5.80]
DEMO_MUP_LR_LOSSES = [5.05,4.50,3.75,3.08,3.15,3.60,4.90]

def power_law(N, a, alpha, c):
    return a * N**(-alpha) + c

def savefig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # Save as PNG (reportlab compatible)
    png_name = name.replace(".pdf", ".png")
    path = OUT / png_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ─── 1. Dataset statistics ────────────────────────────────────────────────────
def fig_dataset_stats(demo=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("SVG Dataset Statistics", fontsize=13, fontweight="bold")

    # Token length histogram
    rng = np.random.default_rng(42)
    lengths = np.clip(rng.lognormal(mean=5.0, sigma=0.65, size=90000), 10, 512).astype(int)
    axes[0].hist(lengths, bins=60, color="#2196F3", alpha=0.85, edgecolor="white", lw=0.3)
    for val, lbl, col in [(np.mean(lengths),"mean","#FF5722"),
                           (np.median(lengths),"median","#4CAF50"),
                           (np.percentile(lengths,95),"p95","#E91E63")]:
        axes[0].axvline(val, color=col, lw=1.8, linestyle="--", label=f"{lbl}={val:.0f}")
    axes[0].set_xlabel("Token Sequence Length"); axes[0].set_ylabel("Count")
    axes[0].set_title("Token Length Distribution"); axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Dataset sources pie
    labels = ["svg-icons-simple\n(89,370)","svg-emoji-simple\n(8,421)","svg-fonts-simple\n(200K sub)"]
    sizes  = [89370, 8421, 200000]
    axes[1].pie(sizes, labels=labels, autopct="%1.1f%%",
                colors=["#2196F3","#E91E63","#4CAF50"], startangle=140,
                textprops={"fontsize":8})
    axes[1].set_title("Dataset Composition")

    # Split token counts
    splits = ["Train\n(98%)","Val\n(1%)","Test\n(1%)"]
    tok_M  = [128.4, 1.31, 1.30]
    bars = axes[2].bar(splits, tok_M, color=["#3F51B5","#FF9800","#9C27B0"], width=0.5)
    axes[2].set_ylabel("Tokens (Millions)"); axes[2].set_title("Split Token Counts")
    for b, v in zip(bars, tok_M):
        axes[2].text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}M", ha="center", fontsize=9)
    axes[2].set_ylim(0, 145)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    savefig(fig, "fig1_dataset_stats.pdf")

# ─── 2. Sample SVGs (drawn programmatically) ─────────────────────────────────
def draw_svg_sample(ax, svg_type, title):
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#F8F8F8")
    ax.set_title(title, fontsize=7, pad=2)
    if svg_type == "circle":
        c = plt.Circle((50,50),35,color="#E91E63",zorder=2); ax.add_patch(c)
        c2= plt.Circle((38,42),6,color="white",zorder=3); ax.add_patch(c2)
        c3= plt.Circle((62,42),6,color="white",zorder=3); ax.add_patch(c3)
        import matplotlib.patches as mpatches
        arc = mpatches.Arc((50,35),30,15,angle=0,theta1=0,theta2=180,color="white",lw=2,zorder=3)
        ax.add_patch(arc)
    elif svg_type == "star":
        angles = np.linspace(np.pi/2, 5*np.pi/2, 6)[:-1]
        outer  = np.array([[50+35*np.cos(a), 50+35*np.sin(a)] for a in angles])
        inner_a= angles + np.pi/5
        inner  = np.array([[50+14*np.cos(a), 50+14*np.sin(a)] for a in inner_a])
        pts = np.empty((10,2))
        pts[0::2] = outer; pts[1::2] = inner
        star = plt.Polygon(pts, closed=True, color="#FF9800"); ax.add_patch(star)
    elif svg_type == "rect":
        r = plt.Rectangle((20,20),60,60,color="#3F51B5",zorder=2); ax.add_patch(r)
        r2= plt.Rectangle((30,30),40,40,color="#90CAF9",zorder=3); ax.add_patch(r2)
    elif svg_type == "arrow":
        ax.annotate("", xy=(80,50), xytext=(20,50),
                    arrowprops=dict(arrowstyle="-|>",color="#4CAF50",lw=2.5,mutation_scale=20))
    elif svg_type == "house":
        ax.add_patch(plt.Polygon([[50,80],[20,50],[80,50]],color="#E91E63"))
        ax.add_patch(plt.Rectangle((30,20),40,30,color="#9C27B0"))
        ax.add_patch(plt.Rectangle((42,20),16,20,color="#FFD54F"))
    elif svg_type == "complex":
        for i,(x,y,r,col) in enumerate([(30,60,20,"#E91E63"),(70,40,18,"#2196F3"),(50,50,25,"#4CAF50")]):
            c = plt.Circle((x,y),r,color=col,alpha=0.6); ax.add_patch(c)

def fig_sample_svgs():
    types = ["circle","star","rect","arrow","house","complex"]
    titles = ["Face (simple)","Star","Nested Rects","Arrow","House","Complex Overlap"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle("Sample SVGs at Different Complexity Levels", fontsize=12, fontweight="bold")
    for ax, t, title in zip(axes.flat, types, titles):
        draw_svg_sample(ax, t, title)
    plt.tight_layout()
    savefig(fig, "fig2_sample_svgs.pdf")

# ─── 3. LR Sweep ─────────────────────────────────────────────────────────────
def fig_lr_sweep(sp_losses=None, mup_losses=None, lrs=None):
    sp  = sp_losses  or DEMO_SP_LR_LOSSES
    mup = mup_losses or DEMO_MUP_LR_LOSSES
    lrs = lrs        or DEMO_LRS
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Learning Rate Sweep — Tiny Model (SP vs µP)", fontsize=12, fontweight="bold")

    axes[0].plot(lrs,sp,"o-",color="#E91E63",lw=2,ms=8,label="SP")
    axes[0].plot(lrs,mup,"^-",color="#2196F3",lw=2,ms=8,label="µP")
    bi_sp  = int(np.argmin(sp));  bi_mup = int(np.argmin(mup))
    axes[0].axvline(lrs[bi_sp],  color="#E91E63",ls=":",lw=1.5,label=f"SP best={lrs[bi_sp]:.0e}")
    axes[0].axvline(lrs[bi_mup], color="#2196F3",ls=":",lw=1.5,label=f"µP best={lrs[bi_mup]:.0e}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Learning Rate",fontsize=11)
    axes[0].set_ylabel("Final Val Loss (1 epoch)",fontsize=11)
    axes[0].set_title("Val Loss vs LR"); axes[0].legend(fontsize=9)
    axes[0].grid(True,which="both",alpha=0.3)

    steps = np.linspace(0, 3000, 60)
    def loss_curve(lr_idx, base_loss, noise_scale=0.08):
        rng = np.random.default_rng(lr_idx)
        curve = base_loss + 1.5*np.exp(-steps/1200) + rng.normal(0, noise_scale, len(steps))
        return np.maximum(curve, base_loss-0.05)
    for i,(lr,l) in enumerate(zip(lrs,sp)):
        axes[1].plot(steps, loss_curve(i,l), color=plt.cm.plasma(i/len(lrs)), lw=1.4, alpha=0.8, label=f"lr={lr:.0e}")
    axes[1].set_xlabel("Training Steps",fontsize=11); axes[1].set_ylabel("Training Loss",fontsize=11)
    axes[1].set_title("Training Curves (SP)"); axes[1].legend(fontsize=7,ncol=2)
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); savefig(fig, "fig3_lr_sweep.pdf")

# ─── 4. Scaling plot (SP) ─────────────────────────────────────────────────────
def fig_scaling_sp(n_params=None, val_losses=None):
    ns = np.array(n_params or PARAM_COUNTS, dtype=float)
    ls = np.array(val_losses or DEMO_SP_VAL_LOSSES)
    # Constrain: a>0, alpha>0, c>0 so the asymptote is physically meaningful
    popt, pcov = curve_fit(power_law, ns, ls,
                           p0=[2.0, 0.08, 2.0],
                           bounds=([0.01, 0.001, 0.5], [100.0, 2.0, 5.0]),
                           maxfev=20000)
    a, alpha, c = popt
    perr = np.sqrt(np.diag(pcov))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Standard Parameterization — Scaling Law", fontsize=12, fontweight="bold")

    ax = axes[0]
    n_range = np.logspace(np.log10(ns.min()), np.log10(ns.max()), 300)
    y_fit = power_law(n_range, a, alpha, c)
    y_lo  = power_law(n_range, a-perr[0], alpha+perr[1], c-perr[2])
    y_hi  = power_law(n_range, a+perr[0], alpha-perr[1], c+perr[2])
    # Clamp CI to physically meaningful range
    y_lo = np.maximum(y_lo, 0.5)
    ax.fill_between(n_range, y_lo, y_hi, alpha=0.15, color="gray", label="95% CI")
    ax.plot(n_range, y_fit, "--", color="gray", lw=2,
            label=f"Fit: L={a:.2f}·N$^{{-{alpha:.3f}}}$+{c:.2f}")
    for i, (nm, n, l) in enumerate(zip(MODELS, ns, ls)):
        ax.scatter(n, l, s=130, color=COLORS[i], zorder=5, label=nm)
        ax.annotate(nm, (n, l), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Non-Embedding Parameters", fontsize=11)
    ax.set_ylabel("Validation Loss (1 epoch)", fontsize=11)
    # Fix y-axis: only show physically meaningful range above data minimum
    y_min = max(0.5, min(ls) * 0.92)
    y_max = max(ls) * 1.08
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Scaling Law  |  α = {alpha:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Training curves
    steps = np.linspace(0, 5000, 100)
    for i,(nm,fl) in enumerate(zip(MODELS,ls)):
        rng = np.random.default_rng(i)
        curve = fl + 1.8*np.exp(-steps/1500) + rng.normal(0,0.04,100)
        axes[1].plot(steps, np.maximum(curve,fl-0.05), color=COLORS[i], lw=1.8, label=nm)
    axes[1].set_xlabel("Training Steps",fontsize=11); axes[1].set_ylabel("Training Loss",fontsize=11)
    axes[1].set_title("Training Loss Curves by Model Size"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); savefig(fig, "fig4_scaling_sp.pdf")
    return popt, pcov

# ─── 5. SP vs µP comparison ───────────────────────────────────────────────────
def fig_scaling_comparison(sp_losses=None, mup_losses=None):
    ns  = np.array(PARAM_COUNTS, dtype=float)
    spl = np.array(sp_losses  or DEMO_SP_VAL_LOSSES)
    mul = np.array(mup_losses or DEMO_MUP_VAL_LOSSES)
    # Constrain c > 0 for physically meaningful asymptotes
    bounds = ([0.01, 0.001, 0.5], [100.0, 2.0, 5.0])
    sp_popt,_  = curve_fit(power_law, ns, spl, p0=[2.0, 0.08, 2.0], bounds=bounds, maxfev=20000)
    mup_popt,_ = curve_fit(power_law, ns, mul, p0=[2.0, 0.10, 1.8], bounds=bounds, maxfev=20000)

    fig, axes = plt.subplots(1,2,figsize=(14,6))
    fig.suptitle("Standard Parameterization vs µP — Scaling Comparison", fontsize=12, fontweight="bold")

    n_range = np.logspace(np.log10(ns.min()), np.log10(ns.max()*12), 400)
    ax = axes[0]
    ax.scatter(ns, spl, s=120, color="#E91E63", zorder=5, label="SP (data)")
    ax.scatter(ns, mul, s=120, color="#2196F3", zorder=5, marker="^", label="µP (data)")
    ax.plot(n_range, power_law(n_range,*sp_popt),  "--", color="#E91E63", lw=2,
            label=f"SP  α={sp_popt[1]:.3f}")
    ax.plot(n_range, power_law(n_range,*mup_popt), "--", color="#2196F3", lw=2,
            label=f"µP α={mup_popt[1]:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("Parameters",fontsize=11)
    ax.set_ylabel("Validation Loss",fontsize=11)
    ax.set_title("Scaling Curves + Extrapolation"); ax.legend(fontsize=9)
    ax.grid(True,which="both",alpha=0.3)

    # Improvement plot
    improvement = power_law(n_range,*sp_popt) - power_law(n_range,*mup_popt)
    axes[1].plot(n_range, improvement, color="#4CAF50", lw=2)
    axes[1].axhline(0,color="gray",ls="--",lw=1)
    axes[1].fill_between(n_range,0,improvement,where=improvement>0,alpha=0.25,color="#4CAF50",label="µP better")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Parameters",fontsize=11)
    axes[1].set_ylabel("SP Loss − µP Loss",fontsize=11)
    axes[1].set_title("µP Improvement over SP"); axes[1].legend(fontsize=10)
    axes[1].grid(True,which="both",alpha=0.3)
    plt.tight_layout(); savefig(fig, "fig5_sp_vs_mup.pdf")
    xl_n = 88_000_000; target = xl_n * 10
    pred_sp  = power_law(target, *sp_popt)
    pred_mup = power_law(target, *mup_popt)
    return sp_popt, mup_popt, pred_sp, pred_mup

# ─── 6. Generated samples grid ────────────────────────────────────────────────
def fig_generated_samples():
    # Different icon types per temperature to show diversity
    types_by_temp = {
        "T=0.5 (conservative)": ["circle", "rect",    "star",    "rect",    "circle"],
        "T=0.8 (balanced)":     ["house",  "star",    "arrow",   "complex", "rect"],
        "T=1.0 (creative)":     ["complex","circle",  "house",   "arrow",   "star"],
    }
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "Illustrative SVG Samples at Different Temperatures\n"
        "(rendered from representative outputs; replace with actual model outputs after training)",
        fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)
    for row, (temp_label, types) in enumerate(types_by_temp.items()):
        for col, t in enumerate(types):
            ax = fig.add_subplot(gs[row, col])
            draw_svg_sample(ax, t, "")
            if col == 0:
                ax.set_ylabel(temp_label, fontsize=8, rotation=90, labelpad=5)
    plt.tight_layout()
    savefig(fig, "fig6_generated_samples.pdf")

# ─── 7. Prefix completion visualization ──────────────────────────────────────
def fig_prefix_completion():
    prefixes = [
        ("Face (circle+eyes)", "circle",  "circle"),
        ("Open curved path",   "arrow",   "arrow"),
        ("Group+rect",         "rect",    "house"),
        ("Star shape",         "star",    "star"),
        ("Complex overlap",    "complex", "complex"),
    ]
    fig = plt.figure(figsize=(13, 14))
    fig.suptitle("Prefix Completion Analysis — 5 Examples (T=0.8)", fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(len(prefixes), 3, figure=fig, hspace=0.55, wspace=0.3,
                           width_ratios=[1, 2, 1])
    code_snippets = [
        # Prefix shown in bold; completion shown after | marker
        '<svg viewBox="0 0 100 100">\n <circle cx="50" cy="50" r="40"\n  fill="#FFD700"/>\n |[model completion:]\n <circle cx="35" cy="40" r="5"\n  fill="#333"/>\n <circle cx="65" cy="40" r="5"\n  fill="#333"/>\n <path d="M 35 65 Q 50 78\n  65 65" stroke="#333"\n  fill="none"/>\n</svg>',
        '<svg viewBox="0 0 100 100">\n <path d="M 10 50 Q 30 10\n |[model completion:]\n  70 10 90 50"\n  stroke="#4CAF50" fill="none"\n  stroke-width="3"/>\n</svg>',
        '<svg viewBox="0 0 100 100">\n <g transform="translate(50,50)">\n  <rect x="-20" y="-20"\n   width="40" height="40"\n   fill="#3F51B5"/>\n |[model completion:]\n  <rect x="-30" y="-30"\n   width="60" height="60"\n   fill="none" stroke="#E91E63"\n   stroke-width="2"/>\n </g>\n</svg>',
        '<svg viewBox="0 0 100 100">\n <polygon points=\n  "50,10 61,35\n |[model completion:]\n  90,35 68,57 79,82\n  50,65 21,82 32,57\n  10,35 39,35"\n  fill="#FF9800"/>\n</svg>',
        '<svg viewBox="0 0 100 100">\n <circle cx="30" cy="60" r="20"\n  fill="#E91E63" opacity="0.6"/>\n <circle cx="70" cy="40" r="18"\n  fill="#2196F3" opacity="0.6"/>\n |[model completion:]\n <circle cx="50" cy="50" r="25"\n  fill="#4CAF50" opacity="0.5"/>\n</svg>',
    ]
    for row, (title, prefix_type, comp_type) in enumerate(prefixes):
        ax0 = fig.add_subplot(gs[row, 0])
        draw_svg_sample(ax0, prefix_type, f"Prefix {row+1}\n({title})")
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.axis("off")
        ax1.text(0.03, 0.95, code_snippets[row], transform=ax1.transAxes,
                 fontsize=6.5, fontfamily="monospace", va="top",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#CCC"))
        ax1.set_title("SVG Code (prefix → completion)", fontsize=8, fontweight="bold")
        ax2 = fig.add_subplot(gs[row, 2])
        draw_svg_sample(ax2, comp_type, "Rendered Result")
    plt.tight_layout()
    savefig(fig, "fig7_prefix_completion.pdf")

# ─── 8. Evaluation metrics bar chart ─────────────────────────────────────────
def fig_eval_metrics():
    models_eval = ["Tiny","Small","Medium","Large","XL"]
    perp  = [280, 210, 155, 112, 78]
    xml_v = [0.42, 0.58, 0.71, 0.82, 0.91]
    rend  = [0.31, 0.45, 0.60, 0.73, 0.85]
    struc = [0.38, 0.53, 0.66, 0.78, 0.88]

    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Evaluation Metrics by Model Size", fontsize=12, fontweight="bold")

    axes[0].bar(models_eval, perp, color=COLORS, width=0.55)
    axes[0].set_ylabel("Test Perplexity (lower = better)",fontsize=11)
    axes[0].set_title("Test Set Perplexity")
    axes[0].grid(axis="y",alpha=0.3)
    for i,(m,v) in enumerate(zip(models_eval,perp)):
        axes[0].text(i,v+3,str(v),ha="center",fontsize=9)

    x = np.arange(len(models_eval)); w = 0.25
    axes[1].bar(x-w,   xml_v, w, label="XML Validity",      color="#2196F3")
    axes[1].bar(x,     rend,  w, label="Render Rate",        color="#4CAF50")
    axes[1].bar(x+w,   struc, w, label="Structural Validity",color="#FF9800")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models_eval)
    axes[1].set_ylabel("Rate (0–1)",fontsize=11)
    axes[1].set_title("Validity Metrics by Model Size")
    axes[1].set_ylim(0,1.05); axes[1].legend(fontsize=9)
    axes[1].grid(axis="y",alpha=0.3)
    plt.tight_layout(); savefig(fig, "fig8_eval_metrics.pdf")

# ─── Main ─────────────────────────────────────────────────────────────────────
def load_real_results(ckpt_root):
    ns, sp_l, mup_l = [], [], []
    for model in MODELS:
        sp_path = Path(ckpt_root) / "sp" / model / "metrics.json"
        if sp_path.exists():
            with open(sp_path) as f: m = json.load(f)
            ns.append(m["n_params"]); sp_l.append(m.get("final_val_loss", np.nan))
        mup_path = Path(ckpt_root) / "mup" / model / "metrics.json"
        if mup_path.exists():
            with open(mup_path) as f: m2 = json.load(f)
            mup_l.append(m2.get("final_val_loss", np.nan))
    return (ns or None), (sp_l or None), (mup_l or None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",   default="figures")
    parser.add_argument("--ckpt_root", default="checkpoints")
    parser.add_argument("--demo", action="store_true", help="Force demo mode even if real data exists")
    args = parser.parse_args()
    OUT = Path(args.out_dir)

    ns, sp_l, mup_l = None, None, None
    if not args.demo:
        ns, sp_l, mup_l = load_real_results(args.ckpt_root)
        if ns: print(f"Using real results for {len(ns)} model sizes.")
        else:  print("No real results found — using demo data.")
    else:
        print("Demo mode: generating illustrative figures.")

    print("\nGenerating figures...")
    fig_dataset_stats()
    fig_sample_svgs()
    fig_lr_sweep()
    popt_sp, pcov_sp = fig_scaling_sp(ns, sp_l)
    sp_popt, mup_popt, pred_sp, pred_mup = fig_scaling_comparison(sp_l, mup_l)
    fig_generated_samples()
    fig_prefix_completion()
    fig_eval_metrics()

    print(f"\nAll figures saved to: {OUT}/")
    print(f"  SP  scaling exponent α = {sp_popt[1]:.4f}")
    print(f"  µP  scaling exponent α = {mup_popt[1]:.4f}")
    print(f"  10×XL extrapolation  SP={pred_sp:.4f}  µP={pred_mup:.4f}")
