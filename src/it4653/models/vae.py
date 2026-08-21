"""Variational Autoencoder with reparameterization trick.

Architecture:
    Encoder:  input  →  μ, log(σ²)
    Reparam trick: z = μ + σ·ε  where ε ~ N(0, I)
    Decoder:  z      →  reconstructed output

Loss:
    L = L_reconstruction + β · KL(q(z|x) ‖ p(z))

Placeholder for implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder.

    Args:
        latent_dim: Size of latent space.
        image_channels: Number of input channels.
        image_size: Spatial size of input images.
        beta: Weight for KL divergence term (default 1.0 for standard VAE).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        image_channels: int = 1,
        image_size: int = 28,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        # TODO: implement Encoder (μ + logvar) + Decoder + reparameterization trick
        raise NotImplementedError("VAE not yet implemented.")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to mean and log-variance.

        Returns:
            mu, logvar:posterior parameters.
        """
        raise NotImplementedError

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z with backprop-friendly gradient flow.

        Why: direct sampling z ~ N(μ, σ²) blocks gradient through μ and σ.
        Trick: re-write as z = μ + σ·ε with ε ~ N(0, I) (non-differentiable noise).
        """
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample to image."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: return reconstruction, mu, logvar.

        Needed for loss computation (reconstruction + KL).
        """
        raise NotImplementedError
