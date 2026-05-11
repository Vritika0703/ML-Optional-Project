#!/usr/bin/env python3
"""
Part 4 (supplemental): Side-by-side prefix completion visualization.
For each prefix, renders:
  [prefix rendered] | [completion SVG code snippet] | [completed SVG rendered]

Spec requirement: "Show the prefix, the model's completion, and the rendered result side by side"

Usage:
    python scripts/04b_prefix_visualization.py \
        --samples_dir results/generated_samples \
        --out_dir results/evaluation \
        --temperature 0.8
"""

import sys
import json
import argparse
import logging
import textwrap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import cairosvg
    from PIL import Image
    import io
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False
    logger.warning("CairoSVG not available — cannot render SVGs")

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def render_svg(svg_str: str, size: int = 200) -> "Image.Image":
    """Render an SVG string to a PIL Image. Returns gray placeholder on failure."""
    if not HAS_CAIRO:
        return Image.new("RGBA", (size, size), (220, 220, 220, 255))
    try:
        png = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"),
                               output_width=size, output_height=size)
        return Image.open(io.BytesIO(png)).convert("RGBA")
    except Exception as e:
        logger.debug(f"Render failed: {e}")
        img = Image.new("RGBA", (size, size), (220, 220, 220, 255))
        return img


def truncate_code(code: str, max_chars: int = 300) -> str:
    """Truncate SVG code for display, adding ellipsis."""
    code = code.strip()
    if len(code) > max_chars:
        return code[:max_chars] + "\n…"
    return code


def create_prefix_completion_grid(samples: dict, temperature_key: str, out_path: str):
    """
    Create a figure with N rows (one per prefix) and 3 columns:
      Col 0: Rendered prefix SVG
      Col 1: SVG completion code (text)
      Col 2: Rendered completed SVG
    """
    if not (HAS_CAIRO and HAS_MPL):
        logger.warning("Missing CairoSVG or matplotlib — cannot create visualization")
        return

    conditional = samples.get("conditional", {})
    prefix_keys = sorted(conditional.keys())
    n_rows = len(prefix_keys)

    if n_rows == 0:
        logger.warning("No conditional samples found")
        return

    fig = plt.figure(figsize=(16, n_rows * 4))
    fig.suptitle(
        f"Prefix Completion Analysis (temperature={temperature_key.replace('temp_', '')})",
        fontsize=14, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.5, wspace=0.3,
                           width_ratios=[1, 2, 1])

    for row, prefix_key in enumerate(prefix_keys):
        data = conditional[prefix_key]
        prefix_svg = data["prefix"]
        completions = data["completions"].get(temperature_key, [])
        if not completions:
            continue
        completed_svg = completions[0]  # show first completion

        # Compute completion-only text (what the model added)
        if completed_svg.startswith(prefix_svg):
            completion_text = completed_svg[len(prefix_svg):]
        else:
            completion_text = completed_svg

        # Col 0: Rendered prefix
        ax0 = fig.add_subplot(gs[row, 0])
        # Prefix might not be valid XML, try to close it for rendering
        prefix_for_render = prefix_svg
        if not prefix_svg.rstrip().endswith(">"):
            prefix_for_render = prefix_svg + '">' + "</svg>"
        prefix_img = render_svg(prefix_for_render)
        ax0.imshow(prefix_img)
        ax0.set_title(f"Prefix {row + 1}", fontsize=10, fontweight="bold")
        ax0.axis("off")

        # Col 1: SVG code
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.axis("off")
        display_code = truncate_code(completion_text, max_chars=400)
        ax1.text(
            0.02, 0.98, display_code,
            transform=ax1.transAxes,
            fontsize=7, fontfamily="monospace",
            verticalalignment="top",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#CCCCCC"),
        )
        ax1.set_title("Model Completion (SVG code)", fontsize=10, fontweight="bold")

        # Col 2: Rendered completion
        ax2 = fig.add_subplot(gs[row, 2])
        completed_img = render_svg(completed_svg)
        ax2.imshow(completed_img)
        ax2.set_title("Rendered Result", fontsize=10, fontweight="bold")
        ax2.axis("off")

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved prefix completion visualization: {out_path}")


def main(args):
    samples_path = Path(args.samples_dir) / "generated_samples.json"
    if not samples_path.exists():
        logger.error(f"samples file not found: {samples_path}")
        return

    with open(samples_path) as f:
        samples = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temperature_key = f"temp_{args.temperature}"
    out_path = out_dir / f"prefix_completion_viz_{temperature_key}.png"

    create_prefix_completion_grid(samples, temperature_key, str(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefix completion side-by-side visualization")
    parser.add_argument("--samples_dir", default="results/generated_samples")
    parser.add_argument("--out_dir", default="results/evaluation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature to visualize (0.5, 0.8, or 1.0)")
    args = parser.parse_args()
    main(args)
