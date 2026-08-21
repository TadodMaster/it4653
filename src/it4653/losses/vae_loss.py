"""VAE loss: reconstruction term + KL divergence.

The Evidence Lower Bound (ELBO):
    L = E_q(z|x)[log p(x|z)]  −  β · KL(q(z|x) ‖ N(0, I))

For Gaussian posterior q(z|x) = N(μ, σ²) and prior p(z) = N(0, I),
the KL has a closed form:

    KL = −½ Σ(1 + log(σ²) − μ² − σ²)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """Combined VAE loss: reconstruction + KL divergence.

    Args:
        reconstruction_loss: "bce" (Binary Cross-Entropy) or "mse" (Mean Squared Error).
        beta: Weight on KL term. 1.0 for standard VAE; >1 for β-VAE.
        reduction: "sum" or "mean" over batch dimension.
    """

    def __init__(
        self,
        reconstruction_loss: str = "bce",
        beta: float = 1.0,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        if reconstruction_loss not in ("bce", "mse"):
            raise ValueError(f"reconstruction_loss must be 'bce' or 'mse', got {reconstruction_loss}")
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction}")

        self.reconstruction_loss = reconstruction_loss
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAE ELBO loss.

        Args:
            recon_x: Decoder output (batch, channels, H, W) in [0, 1].
            x: Original input (batch, channels, H, W) in [0, 1].
            mu: Posterior mean          (batch, latent_dim).
            logvar: Posterior log-variance (batch, latent_dim).

        Returns:
            Scalar ELBO loss value (negative, we minimize this).
        """
        # ---- Reconstruction loss ----
        if self.reconstruction_loss == "bce":
            recon = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
        else:
            recon = F.mse_loss(recon_x, x, reduction=self.reduction)

        # ---- KL divergence (closed form) ----
        # D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        )

        if self.reduction == "sum":
            kl = kl_per_sample.sum()
        else:
            kl = kl_per_sample.mean()

        # ---- Total ELBO loss ----
        # We return the negative ELBO (loss to minimize)
        loss = recon + self.beta * kl
        return loss

    def decompose(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return individual loss components for logging.

        Returns:
            Dict with keys "loss", "recon", "kl".
        """
        if self.reconstruction_loss == "bce":
            recon = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
        else:
            recon = F.mse_loss(recon_x, x, reduction=self.reduction)

        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1,
        )
        kl = kl_per_sample.sum() if self.reduction == "sum" else kl_per_sample.mean()

        loss = recon + self.beta * kl
        return {"loss": loss, "recon": recon, "kl": kl}
