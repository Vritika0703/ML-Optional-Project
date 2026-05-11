#!/usr/bin/env python3
"""
Part 4 (supplemental): Extended best-model training.
Trains the best model configuration for multiple epochs / many tokens
with full hyperparameter control.

Usage:
    python scripts/04a_train_best.py \
        --model xl \
        --data_dir data/processed \
        --out_dir checkpoints/best_model \
        --lr 3e-4 \
        --epochs 3 \
        --dropout 0.1 \
        --weight_decay 0.1 \
        --use_mup
"""

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
from models.transformer import ModelConfig, SVGTransformer, make_mup_model, make_mup_optimizer
from models.data_loader import InfiniteTokenStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cosine_lr_with_warmup(step, warmup_steps, max_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_stream, eval_batches, batch_size, device):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = val_stream.next_batch(batch_size)
        x, y = x.to(device), y.to(device)
        dt = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=dt, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_best(args):
    cfg_path = Path(__file__).parent.parent / "configs" / "model_configs.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["models"][args.model]
    train_cfg = cfg["training"]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device_type = "cuda" if device.type == "cuda" else "cpu"

    with open(Path(args.data_dir) / "dataset_stats.json") as f:
        data_stats = json.load(f)
    vocab_size = data_stats["vocab_size"]

    context_length = args.context_length or train_cfg["context_length"]

    config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=args.dropout,
        bias=model_cfg["bias"],
        use_mup=args.use_mup,
    )

    if args.use_mup:
        model = make_mup_model(config, base_d_model=128).to(device)
        optimizer = make_mup_optimizer(
            model, lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(train_cfg["beta1"], train_cfg["beta2"]),
        )
        logger.info("Using µP parameterization")
    else:
        model = SVGTransformer(config).to(device)
        optimizer = model.configure_optimizers(
            weight_decay=args.weight_decay,
            lr=args.lr,
            betas=(train_cfg["beta1"], train_cfg["beta2"]),
            device_type=device_type,
        )

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters()) - model.transformer.wpe.weight.numel()
    logger.info(f"Best model: {args.model} | Params: {n_params:,} | Epochs: {args.epochs}")

    tokens_per_batch = args.batch_size_tokens or train_cfg["batch_size_tokens"]
    batch_size = max(1, tokens_per_batch // context_length)

    train_data_len = int(np.load(str(Path(args.data_dir) / "train.npy"), mmap_mode="r").shape[0])
    steps_per_epoch = train_data_len // (batch_size * context_length)
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = train_cfg["warmup_iters"]
    min_lr = args.lr * train_cfg["min_lr_ratio"]

    train_stream = InfiniteTokenStream(str(Path(args.data_dir) / "train.npy"), context_length, device)
    val_stream = InfiniteTokenStream(str(Path(args.data_dir) / "val.npy"), context_length, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Total steps: {max_steps:,} | Steps/epoch: {steps_per_epoch:,}")

    metrics = {
        "model": args.model, "n_params": n_params, "lr": args.lr, "epochs": args.epochs,
        "dropout": args.dropout, "weight_decay": args.weight_decay, "use_mup": args.use_mup,
        "train_losses": [], "val_losses": [], "steps": [], "tokens_per_second": [],
    }

    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))
    t0 = time.perf_counter()
    best_val_loss = float("inf")
    eval_interval = max(1, steps_per_epoch // 10)

    for step in range(max_steps):
        lr_now = cosine_lr_with_warmup(step, warmup_steps, max_steps, args.lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

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

        if step % 100 == 0:
            t1 = time.perf_counter()
            tps = (100 * batch_size * context_length) / (t1 - t0) if step > 0 else 0
            epoch_frac = step / steps_per_epoch
            logger.info(
                f"Step {step:6d}/{max_steps} (epoch {epoch_frac:.2f}) | "
                f"loss={loss.item():.4f} | lr={lr_now:.2e} | tok/s={tps:,.0f}"
            )
            metrics["train_losses"].append(loss.item())
            metrics["steps"].append(step)
            metrics["tokens_per_second"].append(tps)
            t0 = t1

        if step % eval_interval == 0:
            val_loss = estimate_loss(model, val_stream, 100, batch_size, device)
            metrics["val_losses"].append({"step": step, "val_loss": val_loss})
            logger.info(f"  >> Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "config": config.__dict__},
                    out_dir / "best_checkpoint.pt"
                )
                logger.info(f"  Saved new best checkpoint (val_loss={val_loss:.4f})")

    metrics["final_val_loss"] = estimate_loss(model, val_stream, 200, batch_size, device)
    metrics["best_val_loss"] = best_val_loss

    torch.save({"model_state": model.state_dict(), "config": config.__dict__},
               out_dir / "final_checkpoint.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nBest model training complete.")
    logger.info(f"  Final val loss : {metrics['final_val_loss']:.4f}")
    logger.info(f"  Best val loss  : {best_val_loss:.4f}")
    logger.info(f"  Checkpoints    : {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended best-model training")
    parser.add_argument("--model", default="xl", choices=["tiny", "small", "medium", "large", "xl"])
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="checkpoints/best_model")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--batch_size_tokens", type=int, default=None)
    parser.add_argument("--use_mup", action="store_true", help="Use µP parameterization")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    train_best(args)
