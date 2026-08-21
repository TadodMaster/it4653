"""Model checkpoint save/load utilities."""

from __future__ import annotations

import os
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict[str, list[float]],
    save_dir: str = "./outputs/checkpoints",
    filename: str = "model.pt",
) -> None:
    """Save model weights + optimizer state + training history + epoch index.

    Format:
        {epoch: N, model_state: dict, optimizer_state: dict, history: dict}
    """
    raise NotImplementedError("Not yet implemented.")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    checkpoint_path: str = "./outputs/checkpoints/model.pt",
) -> tuple[int, dict[str, list[float]]]:
    """Load checkpoint and restore model state.

    Returns:
        (resumed_epoch, history_dict)
    """
    raise NotImplementedError("Not yet implemented.")
