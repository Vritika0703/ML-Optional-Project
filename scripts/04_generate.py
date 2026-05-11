#!/usr/bin/env python3
"""
Part 4: Sample generation from a trained SVG Transformer.
Generates unconditional and prefix-conditioned SVG samples
at multiple temperatures using top-k and top-p sampling.

Usage:
    python scripts/04_generate.py \
        --checkpoint checkpoints/sp/xl/best_checkpoint.pt \
        --tokenizer data/processed/tokenizer.json \
        --out_dir results/generated_samples \
        --n_unconditional 10 \
        --n_conditional 5
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer import ModelConfig, SVGTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Example prefix SVGs for prefix-conditioned generation
SVG_PREFIXES = [
    # Face: circle + one eye
    '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40" fill="#FFD700"/><circle cx="35" cy="40" r="5" fill="#333"/>',
    # Open path (not closed)
    '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M 10 50 Q 30 10',
    # Group with one shape
    '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><g transform="translate(50,50)"><rect x="-20" y="-20" width="40" height="40" fill="#3F51B5"/>',
    # Star outline
    '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><polygon points="50,5 61,35 95,35 68,57',
    # Arrow
    '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><line x1="10" y1="50" x2="70" y2="50" stroke="black" stroke-width="3"/>',
]


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SVGTransformer, ModelConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["config"]
    config = ModelConfig(**cfg_dict)
    model = SVGTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    return model, config


def generate_samples(
    model: SVGTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    device: torch.device,
) -> list[str]:
    """Generate `n_samples` completions from `prompt`."""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    prompt_ids = [bos_id] + tokenizer.encode(prompt).ids if prompt else [bos_id]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    results = []
    for _ in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                prompt_tensor.clone(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_id,
            )
        gen_ids = out[0][len(prompt_ids):].tolist()
        # Decode, stopping at EOS
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        svg_text = prompt + tokenizer.decode(gen_ids)
        results.append(svg_text)
    return results


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model, config = load_model(args.checkpoint, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = {}

    temperatures = [0.5, 0.8, 1.0]

    # ── Unconditional generation ──────────────────────────────────
    logger.info("Generating unconditional samples …")
    uncond_prompt = "<svg"
    unconditional = {}
    for temp in temperatures:
        svgs = generate_samples(
            model, tokenizer,
            prompt=uncond_prompt,
            n_samples=args.n_unconditional,
            max_new_tokens=args.max_new_tokens,
            temperature=temp,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        unconditional[f"temp_{temp}"] = svgs
        logger.info(f"  temp={temp}: generated {len(svgs)} samples")

    all_samples["unconditional"] = unconditional

    # ── Prefix-conditioned generation ─────────────────────────────
    logger.info("Generating prefix-conditioned samples …")
    conditional = {}
    for i, prefix in enumerate(SVG_PREFIXES[:args.n_conditional]):
        cond_samples = {}
        for temp in temperatures:
            svgs = generate_samples(
                model, tokenizer,
                prompt=prefix,
                n_samples=3,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            cond_samples[f"temp_{temp}"] = svgs

        conditional[f"prefix_{i}"] = {
            "prefix": prefix,
            "completions": cond_samples,
        }
        logger.info(f"  Prefix {i} done")

    all_samples["conditional"] = conditional

    # Save raw SVG text
    with open(out_dir / "generated_samples.json", "w") as f:
        json.dump(all_samples, f, indent=2)

    # Save individual SVG files
    svg_count = 0
    for temp_key, svgs in unconditional.items():
        t_dir = out_dir / "unconditional" / temp_key
        t_dir.mkdir(parents=True, exist_ok=True)
        for j, svg in enumerate(svgs):
            (t_dir / f"sample_{j:02d}.svg").write_text(svg, encoding="utf-8")
            svg_count += 1

    for prefix_key, data in conditional.items():
        for temp_key, svgs in data["completions"].items():
            t_dir = out_dir / "conditional" / prefix_key / temp_key
            t_dir.mkdir(parents=True, exist_ok=True)
            (t_dir / "prefix.svg").write_text(data["prefix"], encoding="utf-8")
            for j, svg in enumerate(svgs):
                (t_dir / f"completion_{j:02d}.svg").write_text(svg, encoding="utf-8")
                svg_count += 1

    logger.info(f"Saved {svg_count} SVG files to {out_dir}")
    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVG sample generation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    parser.add_argument("--out_dir", default="results/generated_samples")
    parser.add_argument("--n_unconditional", type=int, default=10)
    parser.add_argument("--n_conditional", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    main(args)
