#!/usr/bin/env python3
"""
run_all.py — Full pipeline orchestrator.
Runs all steps in sequence with configurable options.

Usage (single command to reproduce all results):
    python run_all.py \
        --data_dir data/processed \
        --results_dir results \
        --best_lr 3e-4 \
        --skip_preprocess       # if data already prepared
        --skip_lr_sweep         # if best LR already known
"""

import sys
import json
import logging
import argparse
import subprocess
import shlex
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["tiny", "small", "medium", "large", "xl"]


def run(cmd: str, cwd: str = "."):
    logger.info(f"\n>>> {cmd}")
    ret = subprocess.run(shlex.split(cmd), cwd=cwd)
    if ret.returncode != 0:
        logger.error(f"FAILED: {cmd}")
        sys.exit(ret.returncode)


def main(args):
    root = Path(__file__).parent

    # ── Step 1: Preprocess ───────────────────────────────────────
    if not args.skip_preprocess:
        logger.info("\n" + "="*60 + "\nSTEP 1: Data Preprocessing\n" + "="*60)
        preprocess_cmd = (
            f"{sys.executable} scripts/01_preprocess.py "
            f"--output_dir {args.data_dir} "
            f"--vocab_size {args.vocab_size} "
            f"--max_token_len {args.max_token_len}"
        )
        if args.include_fonts: preprocess_cmd += " --include_fonts"
        if args.include_stack: preprocess_cmd += " --include_stack"
        run(preprocess_cmd, cwd=str(root))
        run(f"{sys.executable} scripts/01b_dataset_stats.py "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.results_dir}", cwd=str(root))
    else:
        logger.info("Skipping preprocessing (--skip_preprocess)")

    # ── Step 2: LR sweep on tiny ─────────────────────────────────
    best_lr = args.best_lr
    if not args.skip_lr_sweep:
        logger.info("\n" + "="*60 + "\nSTEP 2: LR Sweep (Tiny)\n" + "="*60)
        run(f"{sys.executable} scripts/02b_lr_sweep.py "
            f"--data_dir {args.data_dir} "
            f"--out_dir {args.results_dir}/lr_sweep", cwd=str(root))
        # Read best LR
        sweep_summary = Path(args.results_dir) / "lr_sweep" / "lr_sweep_summary.json"
        if sweep_summary.exists():
            with open(sweep_summary) as f:
                s = json.load(f)
            best_lr = min(s.values(), key=lambda x: x["final_val_loss"])["lr"]
            logger.info(f"Best LR from sweep: {best_lr}")
    else:
        logger.info(f"Skipping LR sweep. Using LR = {best_lr}")

    # ── Step 3: SP scaling study ─────────────────────────────────
    if not args.skip_sp_training:
        logger.info("\n" + "="*60 + "\nSTEP 3: SP Scaling Study\n" + "="*60)
        for model in MODELS:
            run(f"{sys.executable} scripts/02_train.py "
                f"--model {model} "
                f"--data_dir {args.data_dir} "
                f"--out_dir checkpoints/sp/{model} "
                f"--lr {best_lr}", cwd=str(root))
        run(f"{sys.executable} scripts/02c_scaling_plot.py "
            f"--checkpoints_dir checkpoints/sp "
            f"--out_dir {args.results_dir} "
            f"--extra_n_params 880000000", cwd=str(root))
    else:
        logger.info("Skipping SP training (--skip_sp_training)")

    # ── Step 4a: µP LR sweep ─────────────────────────────────────
    best_mup_lr = args.best_mup_lr or best_lr  # default: same as SP best
    if not args.skip_mup_lr_sweep:
        logger.info("\n" + "="*60 + "\nSTEP 4a: µP LR Sweep (Tiny)\n" + "="*60)
        run(f"{sys.executable} scripts/03a_lr_sweep_mup.py "
            f"--data_dir {args.data_dir} "
            f"--sp_sweep_dir {args.results_dir}/lr_sweep "
            f"--out_dir {args.results_dir}/lr_sweep_mup", cwd=str(root))
        mup_sweep_summary = Path(args.results_dir) / "lr_sweep_mup" / "mup_lr_sweep_summary.json"
        if mup_sweep_summary.exists():
            with open(mup_sweep_summary) as f:
                ms = json.load(f)
            best_mup_lr = min(ms.values(), key=lambda x: x["final_val_loss"])["lr"]
            logger.info(f"Best µP LR from sweep: {best_mup_lr}")
    else:
        logger.info(f"Skipping µP LR sweep. Using LR = {best_mup_lr}")

    # ── Step 4b: µP scaling study ─────────────────────────────────
    if not args.skip_mup_training:
        logger.info("\n" + "="*60 + "\nSTEP 4: µP Scaling Study\n" + "="*60)
        for model in MODELS:
            run(f"{sys.executable} scripts/03_train_mup.py "
                f"--model {model} "
                f"--data_dir {args.data_dir} "
                f"--out_dir checkpoints/mup/{model} "
                f"--lr {best_lr}", cwd=str(root))
        run(f"{sys.executable} scripts/03b_compare_scaling.py "
            f"--sp_dir checkpoints/sp "
            f"--mup_dir checkpoints/mup "
            f"--out_dir {args.results_dir}", cwd=str(root))
    else:
        logger.info("Skipping µP training (--skip_mup_training)")

    # ── Step 4c: Extended best-model training ────────────────────
    if not args.skip_best_training:
        logger.info("\n" + "="*60 + "\nSTEP 4c: Extended Best-Model Training\n" + "="*60)
        run(f"{sys.executable} scripts/04a_train_best.py "
            f"--model xl "
            f"--data_dir {args.data_dir} "
            f"--out_dir checkpoints/best_model "
            f"--lr {best_lr} "
            f"--epochs 3", cwd=str(root))
    else:
        logger.info("Skipping best-model training (--skip_best_training)")

    # ── Step 5: Generate samples ─────────────────────────────────
    if not args.skip_generation:
        logger.info("\n" + "="*60 + "\nSTEP 5: Generation\n" + "="*60)
        # Use best SP checkpoint (XL if available, else largest trained)
        ckpt = None
        for model in reversed(MODELS):
            candidate = Path(f"checkpoints/sp/{model}/best_checkpoint.pt")
            if candidate.exists():
                ckpt = str(candidate)
                break
        if ckpt:
            run(f"{sys.executable} scripts/04_generate.py "
                f"--checkpoint {ckpt} "
                f"--tokenizer {args.data_dir}/tokenizer.json "
                f"--out_dir {args.results_dir}/generated_samples "
                f"--n_unconditional 10 "
                f"--n_conditional 5", cwd=str(root))
        else:
            logger.warning("No SP checkpoint found, skipping generation.")
    else:
        logger.info("Skipping generation (--skip_generation)")

    # ── Step 6: Evaluate ─────────────────────────────────────────
    if not args.skip_evaluation:
        logger.info("\n" + "="*60 + "\nSTEP 6: Evaluation\n" + "="*60)
        ckpt = None
        for model in reversed(MODELS):
            candidate = Path(f"checkpoints/sp/{model}/best_checkpoint.pt")
            if candidate.exists():
                ckpt = str(candidate)
                break
        if ckpt:
            run(f"{sys.executable} scripts/05_evaluate.py "
                f"--checkpoint {ckpt} "
                f"--tokenizer {args.data_dir}/tokenizer.json "
                f"--test_data {args.data_dir}/test.npy "
                f"--samples_dir {args.results_dir}/generated_samples "
                f"--out_dir {args.results_dir}/evaluation", cwd=str(root))
    else:
        logger.info("Skipping evaluation (--skip_evaluation)")

    # ── Step 7: Prefix completion visualization ───────────────────
    logger.info("\n" + "="*60 + "\nSTEP 7: Prefix Completion Visualization\n" + "="*60)
    for temp in ["0.5", "0.8", "1.0"]:
        run(f"{sys.executable} scripts/04b_prefix_visualization.py "
            f"--samples_dir {args.results_dir}/generated_samples "
            f"--out_dir {args.results_dir}/evaluation "
            f"--temperature {temp}", cwd=str(root))

    logger.info("\n" + "="*60 + "\nAll steps complete!\n" + "="*60)
    logger.info(f"Results are in: {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline orchestrator")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--max_token_len", type=int, default=512)
    parser.add_argument("--best_lr", type=float, default=3e-4)
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_lr_sweep", action="store_true")
    parser.add_argument("--skip_sp_training", action="store_true")
    parser.add_argument("--skip_mup_lr_sweep", action="store_true")
    parser.add_argument("--skip_mup_training", action="store_true")
    parser.add_argument("--skip_best_training", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--include_fonts", action="store_true", help="Include fonts dataset in preprocessing")
    parser.add_argument("--include_stack", action="store_true", help="Include stack dataset in preprocessing")
    parser.add_argument("--best_mup_lr", type=float, default=None,
                        help="Best µP LR (if skipping µP sweep)")
    args = parser.parse_args()

    main(args)
