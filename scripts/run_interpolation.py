#!/usr/bin/env python3
"""Run latent-space interpolation experiment on a trained VAE.

Produces a figure with one row per image pair showing smooth
latent-space transitions from one digit to another.

Usage:
    uv run python scripts/run_interpolation.py \
        --checkpoint outputs/checkpoints/vae/vae.pt \
        --dataset mnist \
        --num-pairs 5 \
        --num-steps 10
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from it4653.experiments.interpolation import run_vae_interpolation


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE latent-space interpolation")
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
        "--num-pairs",
        type=int,
        default=5,
        help="Number of image pairs to interpolate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Interpolation steps (including endpoints)",
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
        help="Directory to save output figure",
    )
    args = parser.parse_args()

    save_path = run_vae_interpolation(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        data_root=args.data_root,
        num_pairs=args.num_pairs,
        num_steps=args.num_steps,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(f"\n[Done] Interpolation figure saved to {save_path}")


if __name__ == "__main__":
    main()
