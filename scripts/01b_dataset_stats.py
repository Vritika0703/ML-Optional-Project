#!/usr/bin/env python3
"""
Part 1 (supplemental): Dataset statistics visualization.
Generates histograms and renders sample SVGs.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import cairosvg
    from PIL import Image
    import io
    HAS_CAIRO = True
except Exception:
    HAS_CAIRO = False


def render_svg_to_pil(svg_str: str):
    """Render SVG to PIL Image via CairoSVG."""
    if not HAS_CAIRO:
        return None
    try:
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=128, output_height=128)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        return None


def plot_token_length_histogram(data_dir: str, output_path: str = None):
    """Plot token length distribution histogram + split sizes."""
    data_dir = Path(data_dir)

    with open(data_dir / "dataset_stats.json") as f:
        stats = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SVG Dataset Statistics", fontsize=14, fontweight="bold")

    # Left: real token length histogram
    hist_path = data_dir / "token_length_histogram.json"
    if hist_path.exists():
        with open(hist_path) as f:
            hist_data = json.load(f)
        counts = np.array(hist_data["counts"])
        edges = np.array(hist_data["bin_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        axes[0].bar(centers, counts, width=(edges[1] - edges[0]) * 0.9,
                    color="#2196F3", alpha=0.8, edgecolor="white", linewidth=0.3)
        # Mark key percentiles
        tl = stats["token_length_stats"]
        for name, val, color in [("mean", tl["mean"], "#FF5722"),
                                   ("p95", tl["p95"], "#E91E63"),
                                   ("median", tl["median"], "#4CAF50")]:
            axes[0].axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{name}={val:.0f}")
        axes[0].set_xlabel("Token Sequence Length", fontsize=11)
        axes[0].set_ylabel("Number of SVGs", fontsize=11)
        axes[0].set_title("Token Length Distribution", fontsize=11)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, axis="y", alpha=0.3)
    else:
        # Fallback: summary stats bar chart
        tl = stats["token_length_stats"]
        labels = ["min", "mean", "median", "p95", "max"]
        values = [tl["min"], tl["mean"], tl["median"], tl["p95"], tl["max"]]
        axes[0].bar(labels, values, color=["#4CAF50", "#2196F3", "#2196F3", "#FF9800", "#F44336"])
        axes[0].set_title("Token Length Summary Stats")
        axes[0].set_ylabel("Tokens")
        for i, v in enumerate(values):
            axes[0].text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)

    # Right: split sizes
    splits = stats["splits"]
    split_names = list(splits.keys())
    token_counts = [splits[s]["tokens"] for s in split_names]
    file_counts = [splits[s]["files"] for s in split_names]

    x = np.arange(len(split_names))
    w = 0.35
    axes[1].bar(x - w/2, file_counts, w, label="Files", color="#9C27B0")
    axes[1].bar(x + w/2, [t // 1000 for t in token_counts], w, label="Tokens (K)", color="#00BCD4")
    axes[1].set_title("Dataset Split Sizes")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(split_names)
    axes[1].legend()
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()



def render_svg_grid(sample_json: str, output_path: str = None, n_per_group: int = 5):
    """Render a grid of sample SVGs from each complexity group."""
    if not HAS_CAIRO:
        print("CairoSVG not available, skipping render grid.")
        return

    with open(sample_json) as f:
        samples = json.load(f)

    groups = list(samples.keys())
    n_cols = n_per_group
    n_rows = len(groups)

    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2.5))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4)

    for row_idx, group in enumerate(groups):
        for col_idx, svg_str in enumerate(samples[group][:n_per_group]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = render_svg_to_pil(svg_str)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=20)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(group, fontsize=10, rotation=90)

    fig.suptitle("Sample SVGs by Complexity", fontsize=14, fontweight="bold")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_token_length_histogram(args.data_dir, str(out / "dataset_statistics.png"))
    render_svg_grid(
        str(Path(args.data_dir) / "sample_svgs.json"),
        str(out / "sample_svgs_grid.png"),
    )
