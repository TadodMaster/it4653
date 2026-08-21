"""Variational Autoencoder with reparameterization trick.

Architecture:
    Encoder:  input  → conv layers → flatten → FC → μ, log(σ²)
    Reparam trick: z = μ + σ·ε  where ε ~ N(0, I)
    Decoder:  z → FC → reshape → conv transpose → reconstructed output

Loss (in vae_loss.py):
    L = L_reconstruction + β · KL(q(z|x) ‖ p(z))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder for image reconstruction and generation.

    The encoder outputs the parameters of a Gaussian posterior q(z|x).
    The decoder maps latent samples back to image space.

    Args:
        latent_dim: Size of the latent space z (e.g. 2, 8, 32, 128).
        image_channels: Number of input channels (1 for MNIST, 3 for RGB).
        image_size: Spatial size of input images (28 for MNIST, 64 for CelebA).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        image_channels: int = 1,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size

        # ---- Encoder ----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Dynamically compute the shape after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, image_size, image_size)
            conv_out = self.encoder_conv(dummy)
            self.conv_shape: tuple[int, ...] = tuple(conv_out.shape[1:])  # (C, H, W)
            self.flat_dim = conv_out.view(1, -1).shape[1]

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # ---- Decoder ----
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.flat_dim),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1] for BCE reconstruction loss
        )

    # ------------------------------------------------------------------ #
    #  Encoder
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input image to mean μ and log-variance log(σ²).

        Args:
            x: Input images (batch, channels, H, W).

        Returns:
            mu: Mean of posterior q(z|x)       (batch, latent_dim)
            logvar: Log-variance of posterior   (batch, latent_dim)
        """
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = self.encoder_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ------------------------------------------------------------------ #
    #  Reparameterization trick
    # ------------------------------------------------------------------ #

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z with backprop-friendly gradients.

        Direct sampling z ~ N(μ, σ²) blocks gradient through μ and σ
        because torch.normal() is non-differentiable w.r.t. its parameters.

        Trick: rewrite z = μ + σ·ε where ε ~ N(0, I) is random noise
        sampled OUTSIDE the computation graph. Now z is a differentiable
        function of μ and σ, enabling gradient flow back to the encoder.

        Args:
            mu: Mean (batch, latent_dim).
            logvar: Log-variance log(σ²) (batch, latent_dim).

        Returns:
            z: Latent sample (batch, latent_dim).
        """
        std = torch.exp(0.5 * logvar)          # σ
        eps = torch.randn_like(std)            # ε ~ N(0, I), no gradient
        z = mu + std * eps                     # z ~ N(μ, σ²), differentiable
        return z

    # ------------------------------------------------------------------ #
    #  Decoder
    # ------------------------------------------------------------------ #

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample z back to an image.

        Args:
            z: Latent vector (batch, latent_dim).

        Returns:
            Reconstructed image (batch, channels, H, W) in [0, 1].
        """
        h = self.decoder_fc(z)
        h = h.view(h.size(0), *self.conv_shape)
        x_recon = self.decoder_conv(h)
        return x_recon

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: reconstruct input and return parameters for loss.

        Returns:
            x_recon: Reconstructed images (batch, channels, H, W)
            mu:      Posterior mean          (batch, latent_dim)
            logvar:  Posterior log-variance   (batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    # ------------------------------------------------------------------ #
    #  Sampling (generation)
    # ------------------------------------------------------------------ #

    def sample(self, num_samples: int, device: str = "cuda") -> torch.Tensor:
        """Generate new images by sampling from the prior p(z) = N(0, I).

        Args:
            num_samples: Number of images to generate.
            device: Device to place tensors on.

        Returns:
            Generated images (num_samples, channels, H, W).
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples
