"""Training loops for VAE and CVAE.

Functions:
    train_vae   : Train VAE with ELBO loss, TensorBoard logging, and checkpointing.
    train_cvae  : Train CVAE (conditional generation) with the same infrastructure.
"""

from __future__ import annotations

import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from it4653.losses.vae_loss import VAELoss
from it4653.utils.checkpoints import save_checkpoint


def train_vae(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    num_epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cuda",
    save_dir: str = "./outputs/checkpoints",
    log_dir: str = "./outputs/logs",
    save_every: int = 10,
) -> dict[str, list[float]]:
    """Train a Variational Autoencoder.

    Logs per-epoch metrics to TensorBoard and saves checkpoints periodically.

    Args:
        model: VAE model instance.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        num_epochs: Number of training epochs.
        lr: Adam learning rate.
        beta: KL divergence weight.
        device: "cuda" or "cpu".
        save_dir: Directory for model checkpoints.
        log_dir: Directory for TensorBoard logs.
        save_every: Save checkpoint every N epochs.

    Returns:
        History dict with keys "loss", "recon", "kl", and optionally "val_loss".
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = VAELoss(reconstruction_loss="bce", beta=beta, reduction="sum")
    writer = SummaryWriter(log_dir=log_dir)

    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}
    if val_loader is not None:
        history["val_loss"] = []

    # Sample a fixed batch for consistent image logging
    sample_batch, _ = next(iter(val_loader or train_loader))
    sample_batch = sample_batch[:8].to(device)

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            bs = x.size(0)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)

            # Loss components (with per-sample averaging inside)
            components = criterion.decompose(recon_x, x, mu, logvar)
            loss = components["loss"] / bs  # per-sample average for gradient

            loss.backward()
            optimizer.step()

            # Accumulate raw totals (will divide by total samples later)
            epoch_loss += components["loss"].item()
            epoch_recon += components["recon"].item()
            epoch_kl += components["kl"].item()
            total_samples += bs

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # Per-sample averages for the epoch
        avg_loss = epoch_loss / total_samples
        avg_recon = epoch_recon / total_samples
        avg_kl = epoch_kl / total_samples

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/recon", avg_recon, epoch)
        writer.add_scalar("train/kl", avg_kl, epoch)

        # ---- Validation ----
        val_str = ""
        if val_loader is not None:
            model.eval()
            val_loss_raw = 0.0
            val_samples = 0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    bs = x.size(0)
                    recon_x, mu, logvar = model(x)
                    components = criterion.decompose(recon_x, x, mu, logvar)
                    val_loss_raw += components["loss"].item()
                    val_samples += bs

            avg_val_loss = val_loss_raw / val_samples
            history["val_loss"].append(avg_val_loss)
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            val_str = f" | val: {avg_val_loss:.4f}"

        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} — "
            f"loss: {avg_loss:.4f} | recon: {avg_recon:.4f} | kl: {avg_kl:.4f}{val_str}"
        )

        # ---- Image logging (every 5 epochs) ----
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Reconstruction comparison
                recon_x, _, _ = model(sample_batch)
                comparison = torch.cat([sample_batch[:4], recon_x[:4]], dim=0)
                grid = torchvision.utils.make_grid(comparison, nrow=4)
                writer.add_image("images/reconstruction", grid, epoch)

                # Random samples from prior N(0, I)
                samples = model.sample(num_samples=16, device=device)
                grid = torchvision.utils.make_grid(samples, nrow=4)
                writer.add_image("images/generated", grid, epoch)

        # ---- Checkpointing ----
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                model, optimizer, epoch + 1, history,
                save_dir=save_dir, filename="vae.pt",
            )

    writer.close()
    return history


def train_cvae(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    num_epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cuda",
    save_dir: str = "./outputs/checkpoints",
    log_dir: str = "./outputs/logs",
    save_every: int = 10,
) -> dict[str, list[float]]:
    """Train a Conditional Variational Autoencoder (CVAE).

    Same training loop as `train_vae` but passes labels through the model.
    Labels are embedded inside the CVAE via one-hot encoding.

    Args:
        model: CVAE model instance.
        train_loader: Training data loader (yields x, labels).
        val_loader: Optional validation data loader.
        num_epochs: Number of training epochs.
        lr: Adam learning rate.
        beta: KL divergence weight.
        device: "cuda" or "cpu".
        save_dir: Directory for model checkpoints.
        log_dir: Directory for TensorBoard logs.
        save_every: Save checkpoint every N epochs.

    Returns:
        History dict with keys "loss", "recon", "kl", and optionally "val_loss".
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = VAELoss(reconstruction_loss="bce", beta=beta, reduction="sum")
    writer = SummaryWriter(log_dir=log_dir)

    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}
    if val_loader is not None:
        history["val_loss"] = []

    # Sample a fixed batch for consistent image logging
    sample_batch, sample_labels = next(iter(val_loader or train_loader))
    sample_batch = sample_batch[:8].to(device)
    sample_labels = sample_labels[:8].to(device)

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for x, labels in pbar:
            x = x.to(device)
            labels = labels.to(device)
            bs = x.size(0)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, labels)

            components = criterion.decompose(recon_x, x, mu, logvar)
            loss = components["loss"] / bs

            loss.backward()
            optimizer.step()

            epoch_loss += components["loss"].item()
            epoch_recon += components["recon"].item()
            epoch_kl += components["kl"].item()
            total_samples += bs

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_loss = epoch_loss / total_samples
        avg_recon = epoch_recon / total_samples
        avg_kl = epoch_kl / total_samples

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/recon", avg_recon, epoch)
        writer.add_scalar("train/kl", avg_kl, epoch)

        # ---- Validation ----
        val_str = ""
        if val_loader is not None:
            model.eval()
            val_loss_raw = 0.0
            val_samples = 0
            with torch.no_grad():
                for x, labels in val_loader:
                    x = x.to(device)
                    labels = labels.to(device)
                    bs = x.size(0)
                    recon_x, mu, logvar = model(x, labels)
                    components = criterion.decompose(recon_x, x, mu, logvar)
                    val_loss_raw += components["loss"].item()
                    val_samples += bs

            avg_val_loss = val_loss_raw / val_samples
            history["val_loss"].append(avg_val_loss)
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            val_str = f" | val: {avg_val_loss:.4f}"

        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} — "
            f"loss: {avg_loss:.4f} | recon: {avg_recon:.4f} | kl: {avg_kl:.4f}{val_str}"
        )

        # ---- Image logging (every 5 epochs) ----
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Conditional reconstruction comparison
                recon_x, _, _ = model(sample_batch, sample_labels)
                comparison = torch.cat([sample_batch[:4], recon_x[:4]], dim=0)
                grid = torchvision.utils.make_grid(comparison, nrow=4)
                writer.add_image("images/reconstruction", grid, epoch)

                # Conditional random samples (use same labels as sample batch for consistency)
                samples = model.sample(labels=sample_labels[:16], num_samples=16, device=device)
                grid = torchvision.utils.make_grid(samples, nrow=4)
                writer.add_image("images/generated", grid, epoch)

        # ---- Checkpointing ----
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                model, optimizer, epoch + 1, history,
                save_dir=save_dir, filename="cvae.pt",
            )

    writer.close()
    return history
