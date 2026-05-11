#!/usr/bin/env python3
"""
Part 1: Data Collection and Preprocessing Pipeline
Downloads SVG datasets from HuggingFace, cleans/normalizes SVGs,
trains a BPE tokenizer, and creates train/val/test splits.
"""

import os
import re
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import lxml.etree as lxml_etree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SVG Cleaning / Normalization
# ─────────────────────────────────────────────

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
EXTRA_WS_RE = re.compile(r"\s+")
FLOAT_RE = re.compile(r"(-?\d+\.\d{2,})")


def round_floats(match: re.Match) -> str:
    """Round coordinate floats to 1 decimal place."""
    return f"{float(match.group(0)):.1f}"


def strip_namespace(svg: str) -> str:
    """Remove XML namespace declarations to keep tokens shorter."""
    svg = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", svg)
    return svg


def normalize_svg(svg: str) -> Optional[str]:
    """
    Clean and normalize a single SVG string.
    Returns None if the SVG should be filtered out.
    """
    # Strip XML comments
    svg = COMMENT_RE.sub("", svg)
    # Strip <?xml ... ?> processing instructions
    svg = re.sub(r"<\?xml[^?]*\?>", "", svg)
    # Normalize whitespace
    svg = EXTRA_WS_RE.sub(" ", svg).strip()
    # Round floats to 1 decimal place to reduce vocabulary
    svg = FLOAT_RE.sub(round_floats, svg)
    # Remove namespace declarations
    svg = strip_namespace(svg)
    return svg


def validate_xml(svg: str) -> bool:
    """Check that the SVG parses as valid XML."""
    try:
        lxml_etree.fromstring(svg.encode("utf-8"))
        return True
    except lxml_etree.XMLSyntaxError:
        return False


def try_render_validate(svg: str) -> bool:
    """
    Optionally validate by attempting to render with CairoSVG.
    Returns True if render succeeds, False otherwise.
    """
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────

DATASETS_BASE = {
    "icons": "starvector/svg-icons-simple",
    "emoji": "starvector/svg-emoji-simple",
}
DATASETS_OPTIONAL = {
    "fonts": "starvector/svg-fonts-simple",   # 2.38 GB — subsample with --fonts_subsample
    "stack": "starvector/svg-stack-simple",   # 3.87 GB
}


def load_hf_dataset(name: str, split: str = "train", max_samples: int = None) -> list[str]:
    """Load SVG strings from a HuggingFace dataset, with optional subsampling."""
    logger.info(f"Loading dataset: {name}" + (f" (max {max_samples:,})" if max_samples else ""))
    ds = load_dataset(name, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
        logger.info(f"  Subsampled to {max_samples:,} from {name}")
    # The SVG text field is typically named 'svg' or 'text'
    svg_col = None
    for col in ["svg", "Svg", "text", "svg_code", "content"]:
        if col in ds.column_names:
            svg_col = col
            break
    if svg_col is None:
        raise ValueError(f"Cannot find SVG column in {ds.column_names}")
    svgs = [row[svg_col] for row in ds]
    logger.info(f"  Loaded {len(svgs):,} SVGs from {name}")
    return svgs


# ─────────────────────────────────────────────
# Main preprocessing
# ─────────────────────────────────────────────

MIN_TRAIN_TOKENS = 100_000_000  # 100M token minimum requirement


def preprocess(
    output_dir: str,
    vocab_size: int = 4096,
    max_token_len: int = 512,
    min_char_len: int = 50,
    validate_render: bool = False,
    seed: int = 42,
    train_frac: float = 0.98,
    val_frac: float = 0.01,
    include_fonts: bool = False,
    fonts_subsample: int = 200_000,
    include_stack: bool = False,
    stack_subsample: int = 100_000,
):
    random.seed(seed)
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw SVGs ──────────────────────────────────────────
    all_svgs: list[str] = []
    datasets_to_load = dict(DATASETS_BASE)

    for key, hf_name in datasets_to_load.items():
        try:
            svgs = load_hf_dataset(hf_name)
            all_svgs.extend(svgs)
        except Exception as e:
            logger.warning(f"Failed to load {hf_name}: {e}")

    if include_fonts:
        ds_fonts = load_hf_dataset("starvector/svg-fonts-simple", max_samples=fonts_subsample)
        all_svgs.extend(ds_fonts)
    if include_stack:
        ds_stack = load_hf_dataset("starvector/svg-stack-simple", max_samples=stack_subsample)
        all_svgs.extend(ds_stack)

    logger.info(f"Total raw SVGs loaded: {len(all_svgs):,}")

    # ── 2. Clean + filter ────────────────────────────────────────
    cleaned: list[str] = []
    stats = {
        "total": len(all_svgs),
        "too_short": 0,
        "invalid_xml": 0,
        "render_fail": 0,
        "passed": 0,
    }

    for svg in tqdm(all_svgs, desc="Cleaning SVGs"):
        norm = normalize_svg(svg)
        if norm is None or len(norm) < min_char_len:
            stats["too_short"] += 1
            continue
        if not validate_xml(norm):
            stats["invalid_xml"] += 1
            continue
        if validate_render and not try_render_validate(norm):
            stats["render_fail"] += 1
            continue
        cleaned.append(norm)

    stats["passed"] = len(cleaned)
    logger.info(f"After cleaning: {len(cleaned):,} SVGs")
    logger.info(f"  Filtering stats: {stats}")

    # ── 3. Train BPE tokenizer ───────────────────────────────────
    logger.info(f"Training BPE tokenizer (vocab_size={vocab_size}) …")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    def batch_iter(texts, batch=1000):
        for i in range(0, len(texts), batch):
            yield texts[i : i + batch]

    tokenizer.train_from_iterator(batch_iter(cleaned), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tok_path = str(out / "tokenizer.json")
    tokenizer.save(tok_path)
    logger.info(f"Saved tokenizer to {tok_path}")

    # ── 4. Filter by token length ────────────────────────────────
    logger.info(f"Filtering by max token length ({max_token_len}) …")
    token_lengths: list[int] = []
    filtered: list[str] = []
    filtered_lengths: list[int] = []

    for svg in tqdm(cleaned, desc="Tokenizing"):
        enc = tokenizer.encode(svg)
        n = len(enc.ids)
        token_lengths.append(n)
        if n <= max_token_len:
            filtered.append(svg)
            filtered_lengths.append(n)

    stats["too_long"] = len(cleaned) - len(filtered)
    logger.info(f"After length filter: {len(filtered):,} SVGs")

    # ── 5. Split ─────────────────────────────────────────────────
    random.shuffle(filtered)
    n = len(filtered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_svgs = filtered[:n_train]
    val_svgs = filtered[n_train : n_train + n_val]
    test_svgs = filtered[n_train + n_val :]

    logger.info(f"Split: train={len(train_svgs):,}  val={len(val_svgs):,}  test={len(test_svgs):,}")

    # ── 6. Encode + save binary token arrays ──────────────────────
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    def encode_split(svgs: list[str], name: str) -> int:
        all_ids: list[int] = []
        for svg in tqdm(svgs, desc=f"Encoding {name}"):
            ids = tokenizer.encode(svg).ids
            all_ids.extend([bos_id] + ids + [eos_id])
        arr = np.array(all_ids, dtype=np.uint16)
        np.save(str(out / f"{name}.npy"), arr)
        logger.info(f"  {name}: {len(arr):,} tokens saved")
        return len(arr)

    train_tokens = encode_split(train_svgs, "train")
    val_tokens = encode_split(val_svgs, "val")
    test_tokens = encode_split(test_svgs, "test")

    # ── 100M token requirement check ─────────────────────────────
    if train_tokens < MIN_TRAIN_TOKENS:
        logger.warning(
            f"\n{'!'*60}\n"
            f"  WARNING: Training set has only {train_tokens:,} tokens.\n"
            f"  The project requires at least {MIN_TRAIN_TOKENS:,} (100M).\n"
            f"  Re-run with --include_fonts or --include_stack to add more data.\n"
            f"{'!'*60}"
        )

    # ── 7. Save dataset statistics ───────────────────────────────
    dataset_stats = {
        "raw_count": stats["total"],
        "after_cleaning": stats["passed"],
        "after_length_filter": len(filtered),
        "filtering": stats,
        "vocab_size": vocab_size,
        "max_token_len": max_token_len,
        "token_length_stats": {
            "min": int(np.min(filtered_lengths)),
            "max": int(np.max(filtered_lengths)),
            "mean": float(np.mean(filtered_lengths)),
            "median": float(np.median(filtered_lengths)),
            "p95": float(np.percentile(filtered_lengths, 95)),
        },
        "splits": {
            "train": {"files": len(train_svgs), "tokens": train_tokens},
            "val": {"files": len(val_svgs), "tokens": val_tokens},
            "test": {"files": len(test_svgs), "tokens": test_tokens},
        },
    }

    with open(out / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    logger.info("Dataset statistics saved.")
    logger.info(f"\n{'='*60}")
    logger.info(f"  Total train tokens : {train_tokens:,}")
    logger.info(f"  Vocab size         : {vocab_size}")
    logger.info(f"  Mean seq len (tok) : {dataset_stats['token_length_stats']['mean']:.1f}")
    logger.info(f"{'='*60}\n")

    # ── 8. Save actual token length histogram data ───────────────
    hist_path = out / "token_length_histogram.json"
    hist_counts, hist_edges = np.histogram(filtered_lengths, bins=50)
    with open(hist_path, "w") as f:
        json.dump({
            "counts": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
        }, f, indent=2)
    logger.info(f"Token length histogram saved to {hist_path}")

    # Save a small sample SVG file for qualitative inspection
    sample_path = out / "sample_svgs.json"
    samples = {
        "short": [s for s in train_svgs if len(s) < 300][:5],
        "medium": [s for s in train_svgs if 300 <= len(s) < 800][:5],
        "long": [s for s in train_svgs if len(s) >= 800][:5],
    }
    with open(sample_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Sample SVGs saved to {sample_path}")

    return dataset_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVG preprocessing pipeline")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=4096, help="BPE vocabulary size")
    parser.add_argument("--max_token_len", type=int, default=512, help="Max token sequence length")
    parser.add_argument("--min_char_len", type=int, default=50, help="Min character length")
    parser.add_argument("--validate_render", action="store_true", help="Validate SVG rendering via CairoSVG")
    parser.add_argument("--seed", type=int, default=42)
    # Extra dataset flags to reach 100M tokens
    parser.add_argument("--include_fonts", action="store_true",
                        help="Include starvector/svg-fonts-simple (subsampled)")
    parser.add_argument("--fonts_subsample", type=int, default=600_000,
                        help="Max SVGs to sample from fonts dataset")
    parser.add_argument("--include_stack", action="store_true",
                        help="Include starvector/svg-stack-simple (subsampled)")
    parser.add_argument("--stack_subsample", type=int, default=300_000,
                        help="Max SVGs to sample from stack dataset")
    args = parser.parse_args()

    preprocess(
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        max_token_len=args.max_token_len,
        min_char_len=args.min_char_len,
        validate_render=args.validate_render,
        seed=args.seed,
        include_fonts=args.include_fonts,
        fonts_subsample=args.fonts_subsample,
        include_stack=args.include_stack,
        stack_subsample=args.stack_subsample,
    )
