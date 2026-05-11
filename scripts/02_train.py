#!/usr/bin/env python3
"""
Part 2: Main Training Script — Scaling Study.
Trains a single transformer model size for one epoch and saves checkpoints + metrics.

Usage:
    python scripts/02_train.py --model tiny --data_dir data/processed --out_dir checkpoints/sp/tiny
    python scripts/02_train.py --model xl   --data_dir data/processed --out_dir checkpoints/sp/xl
"""

import os
import sys
import math
import json
import time
import argparse
import logging
from pathlib import Path

import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer import ModelConfig, SVGTransformer
from models.data_loader import InfiniteTokenStream, make_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────

def cosine_lr_with_warmup(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, val_stream, eval_batches: int, batch_size: int, device) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = val_stream.next_batch(batch_size)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(args):
    # Load config
    cfg_path = Path(__file__).parent.parent / "configs" / "model_configs.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["models"][args.model]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    logger.info(f"Using device: {device}")

    # Build model
    context_length = args.context_length or train_cfg["context_length"]
    with open(Path(args.data_dir) / "dataset_stats.json") as f:
        data_stats = json.load(f)
    vocab_size = data_stats["vocab_size"]

    config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        bias=model_cfg["bias"],
        use_mup=False,
    )
    model = SVGTransformer(config).to(device)

    n_params = model.num_parameters(non_embedding=True)
    logger.info(f"Model: {args.model} | Params (non-emb): {n_params:,}")

    # Compile (PyTorch 2.x)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Optimizer
    lr = args.lr
    min_lr = lr * train_cfg["min_lr_ratio"]
    optimizer = model.configure_optimizers(
        weight_decay=train_cfg["weight_decay"],
        lr=lr,
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        device_type=device_type,
    )

    # Data
    tokens_per_batch = train_cfg["batch_size_tokens"]
    batch_size = max(1, tokens_per_batch // context_length)
    logger.info(f"Batch size: {batch_size} seqs × {context_length} tokens = {batch_size*context_length:,} tok/step")

    train_stream = InfiniteTokenStream(str(Path(args.data_dir) / "train.npy"), context_length, device)
    val_stream = InfiniteTokenStream(str(Path(args.data_dir) / "val.npy"), context_length, device)

    # Steps per epoch
    train_tokens = int(np.load(str(Path(args.data_dir) / "train.npy"), mmap_mode="r").shape[0])
    steps_per_epoch = train_tokens // (batch_size * context_length)
    max_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = train_cfg["warmup_iters"]
    eval_interval = max(1, steps_per_epoch // 20)  # evaluate ~20x per epoch
    eval_batches = 50

    logger.info(f"Steps per epoch: {steps_per_epoch:,} | Total steps: {max_steps:,}")

    # Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training metrics
    metrics = {
        "model": args.model,
        "n_params": n_params,
        "lr": lr,
        "train_losses": [],
        "val_losses": [],
        "steps": [],
        "tokens_seen": [],
        "tokens_per_second": [],
        "gpu_memory_gb": [],
    }

    # ── Training loop ─────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))
    tokens_seen = 0
    t0 = time.perf_counter()
    best_val_loss = float("inf")

    for step in range(max_steps):
        lr_now = cosine_lr_with_warmup(step, warmup_steps, max_steps, lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        x, y = train_stream.next_batch(batch_size)
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += batch_size * context_length

        if step % 50 == 0:
            t1 = time.perf_counter()
            dt = t1 - t0
            tps = (50 * batch_size * context_length) / dt if step > 0 else 0
            mem = torch.cuda.memory_allocated(device) / 1e9 if device_type == "cuda" else 0.0
            logger.info(
                f"Step {step:6d}/{max_steps} | loss={loss.item():.4f} | lr={lr_now:.2e} | "
                f"tok/s={tps:,.0f} | VRAM={mem:.2f}GB | tokens={tokens_seen/1e6:.1f}M"
            )
            metrics["train_losses"].append(loss.item())
            metrics["steps"].append(step)
            metrics["tokens_seen"].append(tokens_seen)
            metrics["tokens_per_second"].append(tps)
            metrics["gpu_memory_gb"].append(mem)
            t0 = t1

        if step % eval_interval == 0:
            val_loss = estimate_loss(model, val_stream, eval_batches, batch_size, device)
            metrics["val_losses"].append({"step": step, "val_loss": val_loss})
            logger.info(f"  >> Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "config": config.__dict__,
                }
                torch.save(ckpt, out_dir / "best_checkpoint.pt")

    # Final val loss
    final_val_loss = estimate_loss(model, val_stream, eval_batches * 2, batch_size, device)
    metrics["final_val_loss"] = final_val_loss
    metrics["best_val_loss"] = best_val_loss
    metrics["wall_clock_seconds"] = time.perf_counter() - (t0 - 1)

    logger.info(f"\nFinal val loss: {final_val_loss:.4f}")

    # Save final checkpoint + metrics
    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, out_dir / "final_checkpoint.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {out_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVG Transformer")
    parser.add_argument("--model", required=True, choices=["tiny", "small", "medium", "large", "xl"])
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f"checkpoints/sp/{args.model}"

    train(args)
