"""Latent space visualization experiments.

Required outputs (per course assignment):
    1. For latent_dim=2: scatter plot of encoded test set colored by label
    2. For latent_dim=2: animate / show trajectory during training
    3. Compare how AE vs VAE organize the 2D latent space

Placeholder for implementation.
"""

from __future__ import annotations


def run_latent_space_visualization(
    model_path: str,
    dataset: str = "mnist",
    latent_dim: int = 2,
    output_dir: str = "./outputs/plots",
) -> None:
    """Generate and save 2D latent space plots."""
    raise NotImplementedError("Not yet implemented.")


def run_latent_dimension_sweep(
    dataset: str = "mnist",
    latent_dims: list[int] = None,  # defaults [2, 8, 32, 128]
    output_dir: str = "./outputs/plots",
) -> None:
    """Train VAE with multiple latent dimensions and compare reconstruction quality.

    Saves a bar chart of final reconstruction loss per latent_dim.
    """
    raise NotImplementedError("Not yet implemented.")
