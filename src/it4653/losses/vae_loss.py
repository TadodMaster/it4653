"""VAE loss: reconstruction term + KL divergence.

Formulation:
    L = E_q(z|x)[log p(x|z)]  −  β · KL(q(z|x) ‖ N(0, I))

The KL term has closed form for Gaussian posterior:
    KL = −½ Σ(1 + log(σ²) − μ² − σ²)

Placeholder for implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VAELoss(nn.Module):
    """Combined VAE loss (reconstruction + KL).

    Args:
        reconstruction_loss: "mse" or "bce" — how to compare input and output
        beta: Weight on KL term (1.0 for standard VAE, >1 for β-VAE)
        reduction: "sum" or "mean" over batch dimension
    """

    def __init__(
        self,
        reconstruction_loss: str = "bce",
        beta: float = 1.0,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        raise NotImplementedError("VAELoss not yet implemented.")

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAE loss.

        Args:
            recon_x: Decoder output (batch, channels, H, W)
            x: Original input (batch, channels, H, W)
            mu: Mean from encoder (batch, latent_dim)
            logvar: Log-variance from encoder (batch, latent_dim)

        Returns:
            Scalar loss value.
        """
        raise NotImplementedError
