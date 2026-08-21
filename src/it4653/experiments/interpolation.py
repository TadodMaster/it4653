"""Interpolation experiment: AE vs VAE latent continuity.

Generate a grid showing:
    Row 1: AE linear interpolation (often discontinuities)
    Row 2: VAE linear interpolation (smooth transitions)

This demonstrates that VAE learns a more continuous latent manifold.

Placeholder for implementation.
"""

from __future__ import annotations


def run_interpolation_experiment(
    ae_ckpt: str,
    vae_ckpt: str,
    dataset: str = "mnist",
    num_pairs: int = 8,
    num_steps: int = 10,
    output_dir: str = "./outputs/plots",
) -> None:
    """Run the interpolation experiment and save side-by-side figure."""
    raise NotImplementedError("Not yet implemented.")
