"""Conditional Variational Autoencoder (CVAE).

Extends VAE by conditioning encoder/decoder on class labels.
The label is embedded and concatenated to both the encoder bottleneck
and the decoder latent input, enabling class-conditional generation.

Architecture:
    Encoder:  x → conv → flatten → concat(label_emb) → FC → μ, log(σ²)
    Reparam:  z = μ + σ·ε
    Decoder:  z → concat(label_emb) → FC → reshape → conv transpose → output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """Conditional VAE for label-conditioned image generation.

    Args:
        latent_dim: Size of latent space z.
        num_classes: Number of conditioning labels (e.g. 10 for MNIST).
        image_channels: Number of input channels (1 for MNIST).
        image_size: Spatial size of input images (28 for MNIST).
        label_embed_dim: Embedding size for labels.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 10,
        image_channels: int = 1,
        image_size: int = 28,
        label_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_channels = image_channels
        self.image_size = image_size
        self.label_embed_dim = label_embed_dim

        # Label embedding: one-hot → learned dense vector
        self.label_embed = nn.Linear(num_classes, label_embed_dim)

        # ---- Encoder ----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Dynamic conv output shape
        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, image_size, image_size)
            conv_out = self.encoder_conv(dummy)
            self.conv_shape: tuple[int, ...] = tuple(conv_out.shape[1:])
            self.flat_dim = conv_out.view(1, -1).shape[1]

        encoder_in_dim = self.flat_dim + label_embed_dim
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_in_dim, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # ---- Decoder ----
        decoder_in_dim = latent_dim + label_embed_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(decoder_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.flat_dim),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    # ------------------------------------------------------------------ #
    #  Label helpers
    # ------------------------------------------------------------------ #

    def _embed_label(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert integer labels to dense embedding vectors.

        Args:
            labels: Integer class labels (batch,).

        Returns:
            Dense label embedding (batch, label_embed_dim).
        """
        # One-hot encoding → learned embedding
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return self.label_embed(one_hot)

    # ------------------------------------------------------------------ #
    #  Encoder
    # ------------------------------------------------------------------ #

    def encode(
        self, x: torch.Tensor, labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input image conditioned on label to μ and log(σ²).

        Args:
            x: Input images (batch, channels, H, W).
            labels: Integer class labels (batch,).

        Returns:
            mu: Mean of posterior q(z|x,y)     (batch, latent_dim)
            logvar: Log-variance of posterior   (batch, latent_dim)
        """
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        lbl = self._embed_label(labels)
        h = torch.cat([h, lbl], dim=1)
        h = self.encoder_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ------------------------------------------------------------------ #
    #  Reparameterization (same as VAE)
    # ------------------------------------------------------------------ #

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z with backprop-friendly gradients.

        Args:
            mu: Mean (batch, latent_dim).
            logvar: Log-variance log(σ²) (batch, latent_dim).

        Returns:
            z: Latent sample (batch, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    # ------------------------------------------------------------------ #
    #  Decoder
    # ------------------------------------------------------------------ #

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Decode latent sample z conditioned on label to an image.

        Args:
            z: Latent vector (batch, latent_dim).
            labels: Integer class labels (batch,).

        Returns:
            Reconstructed image (batch, channels, H, W) in [0, 1].
        """
        lbl = self._embed_label(labels)
        h = torch.cat([z, lbl], dim=1)
        h = self.decoder_fc(h)
        h = h.view(h.size(0), *self.conv_shape)
        x_recon = self.decoder_conv(h)
        return x_recon

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: reconstruct input conditioned on label.

        Args:
            x: Input images (batch, channels, H, W).
            labels: Integer class labels (batch,).

        Returns:
            x_recon: Reconstructed images (batch, channels, H, W)
            mu:      Posterior mean          (batch, latent_dim)
            logvar:  Posterior log-variance   (batch, latent_dim)
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

    # ------------------------------------------------------------------ #
    #  Conditional sampling (generation)
    # ------------------------------------------------------------------ #

    def sample(
        self,
        labels: torch.Tensor | None = None,
        num_samples: int = 64,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate images conditioned on labels.

        If labels is None, random labels are sampled uniformly.

        Args:
            labels: Target class labels (N,) or None for random.
            num_samples: Number of images to generate.
            device: Device to place tensors on.

        Returns:
            Generated images (num_samples, channels, H, W).
        """
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_samples,))
        labels = labels.to(device)

        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z, labels)
        return samples

    def sample_class(
        self, target_class: int, num_samples: int = 64, device: str = "cuda",
    ) -> torch.Tensor:
        """Generate images all belonging to a single target class.

        Args:
            target_class: The class label to condition on.
            num_samples: Number of images to generate.
            device: Device to place tensors on.

        Returns:
            Generated images (num_samples, channels, H, W).
        """
        labels = torch.full((num_samples,), target_class, dtype=torch.long, device=device)
        return self.sample(labels=labels, num_samples=num_samples, device=device)
