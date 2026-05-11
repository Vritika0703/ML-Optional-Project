#!/usr/bin/env python3
"""
Part 4: Evaluation pipeline.
Computes quantitative metrics on generated SVG samples:
  - Perplexity on test set
  - XML validity rate (lxml)
  - SVG render rate (CairoSVG)
  - Structural validity (svg root, viewBox, closed tags)
  - Generates a rendered grid image

Usage:
    python scripts/05_evaluate.py \
        --checkpoint checkpoints/sp/xl/best_checkpoint.pt \
        --tokenizer data/processed/tokenizer.json \
        --test_data data/processed/test.npy \
        --samples_dir results/generated_samples \
        --out_dir results/evaluation
"""

import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import lxml.etree as lxml_etree
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer import ModelConfig, SVGTransformer
from models.data_loader import make_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import cairosvg
    from PIL import Image
    import io
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False
    logger.warning("CairoSVG not available — render rate will be skipped")

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────

def is_valid_xml(svg: str) -> bool:
    try:
        lxml_etree.fromstring(svg.encode("utf-8"))
        return True
    except lxml_etree.XMLSyntaxError:
        return False


def is_renderable(svg: str) -> bool:
    if not HAS_CAIRO:
        return False
    try:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        return True
    except Exception:
        return False


def is_structurally_valid(svg: str) -> bool:
    """Check: has <svg> root, viewBox or width/height, closed tags."""
    try:
        root = lxml_etree.fromstring(svg.encode("utf-8"))
        # Root tag must be svg
        tag = root.tag
        if isinstance(tag, str):
            tag = tag.split("}")[-1]  # strip namespace
        if tag != "svg":
            return False
        # Must have viewBox or width
        attribs = {k.split("}")[-1]: v for k, v in root.attrib.items()}
        has_viewbox = "viewBox" in attribs or "width" in attribs
        return has_viewbox
    except Exception:
        return False


# ─────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, test_data_path: str, context_length: int, batch_size: int, device) -> float:
    """Compute perplexity on test set via average cross-entropy loss."""
    data = np.load(test_data_path, mmap_mode="r")
    n = len(data)
    total_loss = 0.0
    total_tokens = 0
    model.eval()

    for i in range(0, n - context_length - 1, context_length * batch_size):
        batch_x, batch_y = [], []
        for b in range(batch_size):
            start = i + b * context_length
            if start + context_length + 1 > n:
                break
            chunk = torch.from_numpy(data[start : start + context_length + 1].astype(np.int64))
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
        if not batch_x:
            break
        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    return ppl


# ─────────────────────────────────────────────
# Render grid
# ─────────────────────────────────────────────

def render_svg_grid(svgs: list[str], output_path: str, ncols: int = 5, title: str = "Generated SVGs"):
    """Render a grid of SVGs to a PNG file."""
    if not (HAS_CAIRO and HAS_MPL):
        logger.warning("Cannot render grid: missing CairoSVG or matplotlib")
        return

    imgs = []
    for svg in svgs:
        try:
            png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=128, output_height=128)
            img = Image.open(io.BytesIO(png)).convert("RGBA")
        except Exception:
            img = Image.new("RGBA", (128, 128), (240, 240, 240, 255))
        imgs.append(img)

    nrows = math.ceil(len(imgs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs[i])
        else:
            ax.axis("off")
            continue
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved render grid: {output_path}")


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────

def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ModelConfig(**ckpt["config"])
    model = SVGTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded model: {config.d_model}d × {config.n_layers}L")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}

    # 1. Perplexity
    if args.test_data:
        logger.info("Computing test set perplexity …")
        ppl = compute_perplexity(model, args.test_data, config.context_length, batch_size=4, device=device)
        metrics["test_perplexity"] = ppl
        logger.info(f"  Test Perplexity: {ppl:.2f}")

    # 2. Load generated samples
    samples_path = Path(args.samples_dir) / "generated_samples.json"
    if not samples_path.exists():
        logger.warning("No generated_samples.json found. Run 04_generate.py first.")
        all_svgs = []
    else:
        with open(samples_path) as f:
            samples = json.load(f)

        # Flatten all generated SVGs
        all_svgs = []
        for temp_key, svgs in samples.get("unconditional", {}).items():
            all_svgs.extend(svgs)

    if all_svgs:
        logger.info(f"Evaluating {len(all_svgs)} generated SVGs …")

        xml_valid = [is_valid_xml(s) for s in all_svgs]
        struct_valid = [is_structurally_valid(s) for s in all_svgs]
        rendered = [is_renderable(s) for s in all_svgs]

        metrics["n_generated"] = len(all_svgs)
        metrics["xml_validity_rate"] = sum(xml_valid) / len(xml_valid)
        metrics["structural_validity_rate"] = sum(struct_valid) / len(struct_valid)
        metrics["render_rate"] = sum(rendered) / len(rendered) if HAS_CAIRO else None

        logger.info(f"  XML valid:      {metrics['xml_validity_rate']:.1%}")
        logger.info(f"  Structurally valid: {metrics['structural_validity_rate']:.1%}")
        if metrics["render_rate"] is not None:
            logger.info(f"  Render rate:    {metrics['render_rate']:.1%}")

        # Render grids by temperature
        for temp_key, svgs in samples.get("unconditional", {}).items():
            render_svg_grid(
                svgs,
                str(out_dir / f"generated_grid_{temp_key}.png"),
                ncols=5,
                title=f"Unconditional Samples ({temp_key})",
            )

        # Conditional completions grid
        cond_samples = []
        for prefix_key, data in samples.get("conditional", {}).items():
            for temp_key, svgs in data.get("completions", {}).items():
                cond_samples.extend(svgs[:2])
        if cond_samples:
            render_svg_grid(
                cond_samples,
                str(out_dir / "conditional_completions_grid.png"),
                ncols=5,
                title="Prefix-Conditioned Completions",
            )

    # Save metrics
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nEvaluation complete. Results saved to {out_dir}")
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print("="*50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVG model evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    parser.add_argument("--test_data", default="data/processed/test.npy")
    parser.add_argument("--samples_dir", default="results/generated_samples")
    parser.add_argument("--out_dir", default="results/evaluation")
    args = parser.parse_args()

    evaluate(args)
