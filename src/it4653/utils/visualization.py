"""Visualization helpers for VAE evaluation.

Functions:
    plot_loss_curves          : Training/validation loss curves from history dict
    plot_latent_space_2d      : Scatter plot of 2D latent encodings colored by label
    plot_reconstruction_grid  : Side-by-side input vs reconstructed images
    plot_interpolation_grid   : Linear interpolation between two latent points
    save_image_grid           : Save a grid of images to file
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # Headless backend — safe on servers without display


def plot_loss_curves(
    history: dict[str, list[float]],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training loss curves: total loss, reconstruction, KL divergence."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total ELBO loss
    axes[0].plot(history["loss"], color="steelblue", linewidth=1.5, label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], color="orange", linewidth=1.5, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("ELBO Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(history["recon"], color="forestgreen", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].set_title("BCE Reconstruction")
    axes[1].grid(True, alpha=0.3)

    # KL divergence
    axes[2].plot(history["kl"], color="crimson", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title(r"KL(q(z|x) || p(z))")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_latent_space_2d(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    num_samples: int = 5000,
    save_path: str | None = None,
) -> plt.Figure:
    """Encode test set and scatter-plot 2D latent coordinates colored by label.

    For latent_dim > 2: plots the first 2 dimensions of μ (deterministic).
    """
    model = model.to(device)
    model.eval()

    all_z: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        count = 0
        for x, labels in dataloader:
            x = x.to(device)
            mu, _ = model.encode(x)

            # Use first 2 dimensions of μ for plotting
            z = mu[:, :2].cpu().numpy()

            all_z.append(z)
            all_labels.append(labels.numpy())

            count += x.size(0)
            if count >= num_samples:
                break

    all_z = np.concatenate(all_z, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        all_z[:, 0], all_z[:, 1],
        c=all_labels, cmap="tab10", alpha=0.5, s=10,
    )
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_title(f"Latent Space (latent_dim={model.latent_dim})")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Digit Label")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_reconstruction_grid(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str = "cuda",
    num_show: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Display input images side-by-side with their reconstructions."""
    model = model.to(device)
    model.eval()

    images = images[:num_show].to(device)

    with torch.no_grad():
        recon, _, _ = model(images)

    images = images.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, num_show, figsize=(num_show * 1.3, 2.8))

    for i in range(num_show):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)

        # Reconstruction
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Recon", fontsize=10)

    fig.suptitle(f"VAE Reconstructions (latent_dim={model.latent_dim})", fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_interpolation_grid(
    model: torch.nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    num_steps: int = 10,
    device: str = "cuda",
    save_path: str | None = None,
) -> plt.Figure:
    """Linearly interpolate between two latent points and decode each step."""
    model = model.to(device)
    model.eval()

    images = torch.stack([image1, image2]).to(device)

    with torch.no_grad():
        mu, _ = model.encode(images)
        z1, z2 = mu[0], mu[1]

    # Linear interpolation: z(t) = (1-t)·z₁ + t·z₂, t ∈ [0, 1]
    alphas = torch.linspace(0, 1, num_steps, device=device)
    z_interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])

    with torch.no_grad():
        recon_interp = model.decode(z_interp).cpu()

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.2, 1.5))

    for i in range(num_steps):
        axes[i].imshow(recon_interp[i].squeeze(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"{alphas[i].item():.1f}", fontsize=8)

    fig.suptitle(
        f"Linear Interpolation in Latent Space (latent_dim={model.latent_dim})",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    save_path: str | None = None,
    title: str | None = None,
) -> None:
    """Save a grid of images using torchvision and matplotlib."""
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(nrow * 1.2, max(1, len(images) // nrow) * 1.2))
    if grid_np.shape[2] == 1:
        ax.imshow(grid_np.squeeze(), cmap="gray")
    else:
        ax.imshow(grid_np)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
