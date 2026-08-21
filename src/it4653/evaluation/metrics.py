"""Quantitative evaluation metrics for generative models.

Functions:
    compute_fid             : Fréchet Inception Distance (needs InceptionV3)
    compute_inception_score : Inception Score (IS)
    compute_reconstruction_error : MSE / BCE reconstruction loss on test set
    compute_kl_divergence   : KL(q‖p) for VAE evaluation

All functions accept:
    - real_images: Tensor of real samples
    - fake_images: Tensor of generated/reconstructed samples

Placeholder for implementation.
"""

from __future__ import annotations

import torch


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Fréchet Inception Distance.

    Lower is better. Measures distributional similarity between real and generated images
    in InceptionV3 feature space.
    """
    raise NotImplementedError("FID not yet implemented.")


def compute_inception_score(images: torch.Tensor, splits: int = 10) -> tuple[float, float]:
    """Inception Score.

    Higher is better. Measures quality and diversity of generated images.

    Returns:
        (mean, std) of IS across splits.
    """
    raise NotImplementedError("IS not yet implemented.")


def compute_reconstruction_error(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> float:
    """Mean reconstruction error (MSE or BCE) on a test set."""
    raise NotImplementedError("Reconstruction error not yet implemented.")


def compare_models(
    ae_ckpt: str,
    vae_ckpt: str,
    gan_ckpt: str,
    test_loader: torch.utils.data.DataLoader,
    output_dir: str = "./outputs/plots",
) -> None:
    """Run all comparison experiments:
        - Generate side-by-side image grids at same epoch
        - Compute FID / IS table
        - Save results as CSV + figure
    """
    raise NotImplementedError("Model comparison not yet implemented.")
