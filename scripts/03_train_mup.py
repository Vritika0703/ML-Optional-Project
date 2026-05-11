#!/usr/bin/env python3
"""
Part 3: µP Training Script — identical to 02_train.py but uses
µP parameterization (mup package) and MuAdamW optimizer.

Usage:
    python scripts/03_train_mup.py --model tiny --data_dir data/processed --lr 3e-4
    python scripts/03_train_mup.py --model xl   --data_dir data/processed --lr 3e-4  # same LR transfers!
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


def cosine_lr_with_warmup(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    import math
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_stream, eval_batches: int, batch_size: int, device) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = val_stream.next_batch(batch_size)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_mup(args):
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
    logger.info(f"Device: {device} | µP mode: ON")

    # Load dataset stats
    with open(Path(args.data_dir) / "dataset_stats.json") as f:
        data_stats = json.load(f)
    vocab_size = data_stats["vocab_size"]

    context_length = args.context_length or train_cfg["context_length"]

    # Build µP model
    config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        bias=model_cfg["bias"],
        use_mup=True,
    )

    model = make_mup_model(config, base_d_model=128)  # tiny is the base
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) - model.transformer.wpe.weight.numel()
    logger.info(f"µP Model: {args.model} | Params (non-emb): {n_params:,}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    lr = args.lr
    min_lr = lr * train_cfg["min_lr_ratio"]
    optimizer = make_mup_optimizer(
        model, lr=lr,
        weight_decay=train_cfg["weight_decay"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )

    tokens_per_batch = train_cfg["batch_size_tokens"]
    batch_size = max(1, tokens_per_batch // context_length)

    train_data = np.load(str(Path(args.data_dir) / "train.npy"), mmap_mode="r")
    train_tokens = train_data.shape[0]
    steps_per_epoch = train_tokens // (batch_size * context_length)
    max_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = train_cfg["warmup_iters"]
    eval_interval = max(1, steps_per_epoch // 20)

    train_stream = InfiniteTokenStream(str(Path(args.data_dir) / "train.npy"), context_length, device)
    val_stream = InfiniteTokenStream(str(Path(args.data_dir) / "val.npy"), context_length, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": args.model,
        "n_params": n_params,
        "lr": lr,
        "parameterization": "mup",
        "train_losses": [],
        "val_losses": [],
        "steps": [],
        "tokens_seen": [],
        "tokens_per_second": [],
        "gpu_memory_gb": [],
    }

    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))
    tokens_seen = 0
    t0 = time.perf_counter()
    best_val_loss = float("inf")

    for step in range(max_steps):
        lr_now = cosine_lr_with_warmup(step, warmup_steps, max_steps, lr, min_lr)
        # Note: for µP, the LR scaling is handled per-parameter by MuAdamW;
        # we still schedule the global LR with cosine decay
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
                f"[µP] Step {step:6d}/{max_steps} | loss={loss.item():.4f} | lr={lr_now:.2e} | "
                f"tok/s={tps:,.0f} | VRAM={mem:.2f}GB"
            )
            metrics["train_losses"].append(loss.item())
            metrics["steps"].append(step)
            metrics["tokens_seen"].append(tokens_seen)
            metrics["tokens_per_second"].append(tps)
            metrics["gpu_memory_gb"].append(mem)
            t0 = t1

        if step % eval_interval == 0:
            val_loss = estimate_loss(model, val_stream, 50, batch_size, device)
            metrics["val_losses"].append({"step": step, "val_loss": val_loss})
            logger.info(f"  >> [µP] Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "config": config.__dict__},
                    out_dir / "best_checkpoint.pt"
                )

    final_val_loss = estimate_loss(model, val_stream, 100, batch_size, device)
    metrics["final_val_loss"] = final_val_loss
    metrics["best_val_loss"] = best_val_loss

    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, out_dir / "final_checkpoint.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[µP] Final val loss: {final_val_loss:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="µP Training")
    parser.add_argument("--model", required=True, choices=["tiny", "small", "medium", "large", "xl"])
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f"checkpoints/mup/{args.model}"

    train_mup(args)
