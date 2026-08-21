#!/usr/bin/env python3
"""Train a Conditional VAE (CVAE) on MNIST or Fashion-MNIST.

Usage:
    uv run python scripts/train_cvae.py --config configs/cvae.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch

from it4653.data.datasets import get_mnist_loaders
from it4653.models.cvae import CVAE
from it4653.training.trainers import train_cvae
from it4653.utils.config import load_config
from it4653.utils.visualization import plot_loss_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conditional VAE")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cvae.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Override latent_dim from config",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    m_cfg = config["model"]
    ds_cfg = config["dataset"]
    t_cfg = config["training"]
    l_cfg = config["logging"]

    # Overrides
    if args.latent_dim is not None:
        m_cfg["latent_dim"] = args.latent_dim
        print(f"[Override] latent_dim = {args.latent_dim}")

    # Device
    device = t_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[!] CUDA not available, falling back to CPU")

    # Data loaders
    train_loader, val_loader = get_mnist_loaders(
        dataset=ds_cfg["name"],
        data_root=ds_cfg["data_root"],
        batch_size=ds_cfg["batch_size"],
        image_size=ds_cfg["image_size"],
        num_workers=0,  # safer on Windows
    )

    # Model
    model = CVAE(
        latent_dim=m_cfg["latent_dim"],
        num_classes=m_cfg["num_classes"],
        image_channels=m_cfg["image_channels"],
        image_size=m_cfg["image_size"],
        label_embed_dim=m_cfg.get("label_embed_dim", 8),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{'='*50}")
    print(f"  Model:     CVAE(latent_dim={m_cfg['latent_dim']}, num_classes={m_cfg['num_classes']})")
    print(f"  Dataset:   {ds_cfg['name']}")
    print(f"  Batch:     {ds_cfg['batch_size']}")
    print(f"  Epochs:    {t_cfg['num_epochs']}")
    print(f"  LR:        {t_cfg['lr']}")
    print(f"  Beta:      {m_cfg.get('beta', 1.0)}")
    print(f"  Device:    {device}")
    print(f"  Params:    {n_params:,}")
    print(f"{'='*50}\n")

    # Train
    history = train_cvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=t_cfg["num_epochs"],
        lr=t_cfg["lr"],
        beta=m_cfg.get("beta", 1.0),
        device=device,
        save_dir=l_cfg["checkpoint_dir"],
        log_dir=l_cfg["log_dir"],
        save_every=t_cfg["save_every"],
    )

    # Final summary
    print(f"\n{'='*50}")
    print(f"  Training complete!")
    print(f"  Final train loss: {history['loss'][-1]:.4f}")
    print(f"  Final recon loss: {history['recon'][-1]:.4f}")
    print(f"  Final KL loss:    {history['kl'][-1]:.4f}")
    if "val_loss" in history:
        print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")
    print(f"{'='*50}")

    # Save loss curves
    plot_dir = Path(l_cfg["log_dir"]).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(history, save_path=str(plot_dir / "loss_curves.png"))
    print(f"\n[Plots] Saved to {plot_dir}")

    # Launch tensorboard hint
    print(f"\n[TensorBoard] Run:  tensorboard --logdir={l_cfg['log_dir']}")


if __name__ == "__main__":
    main()
