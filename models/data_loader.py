#!/usr/bin/env python3
"""
DataLoader utilities for the SVG token datasets.
Yields batches of (input, target) token tensors from pre-tokenized .npy files.
"""

import numpy as np
import torch
from pathlib import Path


class SVGDataset(torch.utils.data.Dataset):
    """
    Memory-mapped dataset over a pre-tokenized token array (.npy).
    Each example is a random context_length window.
    """

    def __init__(self, data_path: str, context_length: int):
        self.data = np.load(data_path, mmap_mode="r")
        self.context_length = context_length
        # Number of valid windows
        self.n = len(self.data) - context_length

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx: int):
        chunk = torch.from_numpy(
            self.data[idx : idx + self.context_length + 1].astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def make_dataloader(
    data_path: str,
    context_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    ds = SVGDataset(data_path, context_length)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class InfiniteTokenStream:
    """
    Fast data iterator that reads flat token arrays and
    yields (x, y) tensors in token-budget-based batches.
    Wraps around when the end of the file is reached.
    Used for the main training loop (tokens-per-batch mode).
    """

    def __init__(self, data_path: str, context_length: int, device: torch.device):
        self.data = np.load(data_path, mmap_mode="r")
        self.n = len(self.data)
        self.context_length = context_length
        self.device = device
        self.pos = 0

    def next_batch(self, batch_size: int):
        T = self.context_length
        xs, ys = [], []
        for _ in range(batch_size):
            if self.pos + T + 1 > self.n:
                self.pos = 0  # wrap
            chunk = torch.from_numpy(
                self.data[self.pos : self.pos + T + 1].astype(np.int64)
            )
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.pos += T + 1
        x = torch.stack(xs).to(self.device)
        y = torch.stack(ys).to(self.device)
        return x, y

    def tokens_remaining(self) -> int:
        return self.n - self.pos
