"""Memory-bounded author operations and recoverable epoch checkpoints."""

import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def bounded_attention(self, query, key, value, mask=None, dropout=None):
    """Keep the author's feature-specific softmax and bound the query axis."""
    if dropout is not None:
        raise ValueError("The verified CPU attention supports the author's dropout=None path")

    def compute(part, keys, values):
        scores = torch.matmul(part, keys.transpose(-2, -1)) / (part.shape[-1] ** 0.5)
        scores = scores.unsqueeze(-1).expand(*scores.shape, values.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        weights = torch.softmax(scores, dim=-2)
        return torch.sum(weights * values.unsqueeze(-3), -2)

    output = []
    for part in query.split(8, dim=-2):
        if torch.is_grad_enabled():
            output.append(checkpoint(compute, part, key, value, use_reentrant=False))
        else:
            output.append(compute(part, key, value))
    return torch.cat(output, dim=-2), None


def save_epoch(path, *, epoch, models, optimizer, scheduler, metadata):
    """Save enough state to restart at the next epoch without losing the RNG."""
    path = Path(path)
    state = {
        "epoch": epoch, "models": {name: model.state_dict() for name, model in models.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metadata": metadata, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    staged = path.with_suffix(".pending")
    torch.save(state, staged)
    os.replace(staged, path)


def restore_epoch(path, *, models, optimizer, scheduler):
    state = torch.load(path, map_location="cpu")
    for name, model in models.items():
        model.load_state_dict(state["models"][name])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    return state
