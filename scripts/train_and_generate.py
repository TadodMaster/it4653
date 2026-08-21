#!/usr/bin/env python3
"""Train a VAE and immediately generate images from it.

A single command for the full workflow:
    uv run python scripts/train_and_generate.py

Optional overrides:
    uv run python scripts/train_and_generate.py --latent-dim 32 --epochs 50 --samples 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch
import torchvision

from it4653.data.datasets import get_mnist_loaders
from it4653.models.vae import VAE
from it4653.training.trainers import train_vae
from it4653.utils.checkpoints import load_checkpoint
from it4653.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE + Generate images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vae.yaml",
        help="Config YAML file",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Override latent dimension (default from config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (default from config)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of images to generate after training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "fashion-mnist"],
        help="Override dataset (default from config)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    m_cfg = config["model"]
    ds_cfg = config["dataset"]
    t_cfg = config["training"]
    l_cfg = config["logging"]

    # Apply overrides
    if args.latent_dim is not None:
        m_cfg["latent_dim"] = args.latent_dim
    if args.epochs is not None:
        t_cfg["num_epochs"] = args.epochs
    if args.dataset is not None:
        ds_cfg["name"] = args.dataset

    # Device
    device = t_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Output dirs
    checkpoint_dir = Path(l_cfg["checkpoint_dir"])
    image_dir = Path(l_cfg.get("image_dir", "./outputs/generated_images/vae"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  VAE — Train + Generate")
    print(f"  Dataset:   {ds_cfg['name']}")
    print(f"  Latent:    {m_cfg['latent_dim']}")
    print(f"  Epochs:    {t_cfg['num_epochs']}")
    print(f"  Samples:   {args.samples}")
    print(f"  Device:    {device}")
    print("=" * 55)

    # Data loaders
    train_loader, val_loader = get_mnist_loaders(
        dataset=ds_cfg["name"],
        data_root=ds_cfg["data_root"],
        batch_size=ds_cfg["batch_size"],
        image_size=ds_cfg["image_size"],
        num_workers=0,
    )

    # Model
    model = VAE(
        latent_dim=m_cfg["latent_dim"],
        image_channels=m_cfg["image_channels"],
        image_size=m_cfg["image_size"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params:    {n_params:,}\n")

    # Train
    history = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=t_cfg["num_epochs"],
        lr=t_cfg["lr"],
        beta=m_cfg.get("beta", 1.0),
        device=device,
        save_dir=str(checkpoint_dir),
        log_dir=l_cfg["log_dir"],
        save_every=t_cfg["save_every"],
    )

    print(f"\n{'='*55}")
    print("  Training complete!")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"{'='*55}")

    # Load best checkpoint and generate images
    print("\n[*] Generating images...")
    ckpt_path = checkpoint_dir / "vae.pt"
    if ckpt_path.exists():
        print(f"    Loading checkpoint: {ckpt_path}")
        model = VAE(
            latent_dim=m_cfg["latent_dim"],
            image_channels=m_cfg["image_channels"],
            image_size=m_cfg["image_size"],
        )
        load_checkpoint(str(ckpt_path), model, device=device)
        model = model.to(device).eval()
    else:
        print(f"    [!] Checkpoint not found at {ckpt_path}, using in-memory model")
        model = model.to(device).eval()

    with torch.no_grad():
        # Generated from random z ~ N(0, I)
        samples = model.sample(num_samples=args.samples, device=device)

    # Save as grid
    grid_path = image_dir / "generated_grid.png"
    grid = torchvision.utils.make_grid(samples, nrow=8, padding=2)
    torchvision.utils.save_image(grid, str(grid_path))
    print(f"    Saved grid: {grid_path}")

    # Save individual images
    for i, img in enumerate(samples[:10]):
        single_path = image_dir / f"generated_{i}.png"
        torchvision.utils.save_image(img, str(single_path))
    print(f"    Saved: {image_dir}/generated_0.png .. generated_9.png")

    # Also save the interpolation grid if using same trained model
    print("\n[*] Generating interpolation...")
    _, test_loader = get_mnist_loaders(
        dataset=ds_cfg["name"],
        data_root=ds_cfg["data_root"],
        batch_size=2,
        image_size=ds_cfg["image_size"],
        num_workers=0,
    )
    x_test, _ = next(iter(test_loader))
    x1, x2 = x_test[0].to(device), x_test[1].to(device)

    # Encode both images
    with torch.no_grad():
        mu, _ = model.encode(torch.stack([x1, x2]))
        z1, z2 = mu[0], mu[1]
        alphas = torch.linspace(0, 1, 10, device=device)
        z_interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
        recon_interp = model.decode(z_interp)

    interp_path = image_dir / "interpolation.png"
    interp_grid = torchvision.utils.make_grid(recon_interp, nrow=10, padding=2)
    torchvision.utils.save_image(interp_grid, str(interp_path))
    print(f"    Saved interpolation: {interp_path}")

    print(f"\n{'='*55}")
    print("  ALL DONE")
    print(f"  Checkpoint:  {ckpt_path}")
    print(f"  Generated:     {image_dir}")
    print(f"  TensorBoard:   tensorboard --logdir={l_cfg['log_dir']}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
