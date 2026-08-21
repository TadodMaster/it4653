#!/usr/bin/env python3
"""Load a trained VAE and visualize its latent space.

Produces:
1. A 2D scatter plot of latent encodings colored by digit label
2. Latent coordinate histograms per class (optional)

Usage:
    uv run python scripts/visualize_latent_spaces.py \
        --checkpoint outputs/checkpoints/vae/vae.pt \
        --dataset mnist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch

from it4653.data.datasets import get_mnist_loaders
from it4653.models.vae import VAE
from it4653.utils.checkpoints import load_checkpoint
from it4653.utils.visualization import plot_latent_space_2d, plot_reconstruction_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE latent-space visualization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to VAE checkpoint (.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion-mnist"],
        help="Dataset used for training",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of test samples to encode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/plots",
        help="Directory to save output figures",
    )
    args = parser.parse_args()

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[!] CUDA not available, falling back to CPU")

    # Load model (latent_dim will be overridden by checkpoint weights)
    model = VAE(latent_dim=32, image_channels=1, image_size=28)
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    # Data loaders
    _, test_loader = get_mnist_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Latent space scatter plot
    print("[*] Generating latent space scatter plot...")
    plot_path = out_dir / "latent_space.png"
    fig = plot_latent_space_2d(
        model, test_loader, device=device,
        num_samples=args.num_samples, save_path=str(plot_path),
    )
    print(f"[Done] Saved to {plot_path}")

    # 2. Reconstruction grid (first 8 test images)
    print("[*] Generating reconstruction grid...")
    recon_path = out_dir / "reconstructions.png"
    batch, _ = next(iter(test_loader))
    fig = plot_reconstruction_grid(
        model, batch, device=device, num_show=8, save_path=str(recon_path),
    )
    print(f"[Done] Saved to {recon_path}")

    # 3. Interpolation grid (two random test images)
    print("[*] Generating interpolation grid...")
    interp_path = out_dir / "interpolation.png"
    from it4653.utils.visualization import plot_interpolation_grid
    fig = plot_interpolation_grid(
        model, batch[0], batch[1], num_steps=10, device=device, save_path=str(interp_path),
    )
    print(f"[Done] Saved to {interp_path}")

    print(f"\n{'='*50}")
    print(f"  All visualizations saved to {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
