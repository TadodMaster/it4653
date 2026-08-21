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
    filename: str = "vae.pt",
) -> None:
    """Save model weights, optimizer state, and training history.

    Checkpoint format::

        {
            "epoch": int,
            "model_state_dict": dict,
            "optimizer_state_dict": dict,
            "history": dict,
        }

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch number (1-based).
        history: Training history dict.
        save_dir: Directory to save checkpoints.
        filename: Checkpoint filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, path)
    # print(f"[Checkpoint] Saved to {path} (epoch {epoch})")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda",
) -> tuple[int, dict[str, list[float]]]:
    """Load checkpoint and restore model (and optionally optimizer) state.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load weights into (modified in-place).
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors onto.

    Returns:
        (last_epoch, history): The epoch at which training was saved,
        and the training history dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    history = checkpoint.get("history", {})
    # print(f"[Checkpoint] Loaded from {checkpoint_path} (epoch {epoch})")

    return epoch, history
