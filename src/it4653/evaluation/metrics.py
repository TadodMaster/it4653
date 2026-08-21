"""Quantitative evaluation metrics for VAE.

Functions:
    compute_reconstruction_error : MSE / BCE reconstruction loss on test set
    compute_vae_elbo               : ELBO (recon + KL) on test set
    compute_active_units           : Count latent dimensions with KL > threshold

Note:
    FID and Inception Score require `torchvision.models.inception_v3`,
    which is heavy. Kept as stubs for optional future use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from it4653.losses.vae_loss import VAELoss


def compute_reconstruction_error(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    reconstruction_loss: str = "bce",
    device: str = "cuda",
) -> dict[str, float]:
    """Compute average reconstruction error on test set.

    Args:
        model: VAE or CVAE model.
        test_loader: Test data loader.
        reconstruction_loss: "bce" or "mse".
        device: "cuda" or "cpu".

    Returns:
        Dict with keys "recon_loss", "mse", "mae".
    """
    model = model.to(device)
    model.eval()

    total_recon = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                x, labels = batch
                x = x.to(device)
                labels = labels.to(device)
                recon, mu, logvar = model(x, labels)
            else:
                x = batch[0].to(device)
                recon, mu, logvar = model(x)

            bs = x.size(0)

            if reconstruction_loss == "bce":
                recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
            else:
                recon_loss = F.mse_loss(recon, x, reduction="sum")

            mse = F.mse_loss(recon, x, reduction="sum")
            mae = F.l1_loss(recon, x, reduction="sum")

            total_recon += recon_loss.item()
            total_mse += mse.item()
            total_mae += mae.item()
            total_samples += bs

    return {
        "recon_loss": total_recon / total_samples,
        "mse": total_mse / total_samples,
        "mae": total_mae / total_samples,
    }


def compute_vae_elbo(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    beta: float = 1.0,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute average ELBO, reconstruction, and KL on test set.

    Args:
        model: VAE or CVAE model.
        test_loader: Test data loader.
        beta: KL weight.
        device: "cuda" or "cpu".

    Returns:
        Dict with keys "loss", "recon", "kl" (all per-sample averages).
    """
    model = model.to(device)
    model.eval()

    criterion = VAELoss(reconstruction_loss="bce", beta=beta, reduction="sum")
    total_loss = total_recon = total_kl = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                x, labels = batch
                x = x.to(device)
                labels = labels.to(device)
                recon, mu, logvar = model(x, labels)
            else:
                x = batch[0].to(device)
                recon, mu, logvar = model(x)

            bs = x.size(0)
            comps = criterion.decompose(recon, x, mu, logvar)
            total_loss += comps["loss"].item()
            total_recon += comps["recon"].item()
            total_kl += comps["kl"].item()
            total_samples += bs

    return {
        "loss": total_loss / total_samples,
        "recon": total_recon / total_samples,
        "kl": total_kl / total_samples,
    }


def compute_active_units(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    threshold: float = 0.01,
    device: str = "cuda",
) -> dict[str, int | torch.Tensor]:
    """Count "active" latent units (dimensions with KL > threshold).

    Inactive units have collapsed to the prior (KL ≈ 0), which hurts
    generative quality. This metric is useful for tuning latent_dim.

    Args:
        model: VAE model.
        test_loader: Test data loader.
        threshold: KL threshold per dimension.
        device: "cuda" or "cpu".

    Returns:
        Dict with "active" (int), "total" (int), and "kl_per_dim" (tensor).
    """
    model = model.to(device)
    model.eval()

    all_kl = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                x, labels = batch
                x = x.to(device)
                labels = labels.to(device)
                mu, logvar = model.encode(x, labels)
            else:
                x = batch[0].to(device)
                mu, logvar = model.encode(x)

            # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
            kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            all_kl.append(kl_dim)

    all_kl = torch.cat(all_kl, dim=0)  # (N, latent_dim)
    mean_kl = all_kl.mean(dim=0)       # (latent_dim,)
    active = (mean_kl > threshold).sum().item()

    return {
        "active": active,
        "total": mean_kl.numel(),
        "kl_per_dim": mean_kl.cpu(),
    }


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Fréchet Inception Distance.

    Lower is better. Measures distributional similarity between real
    and generated images in InceptionV3 feature space.

    Requires ``torchvision.models.inception_v3`` (not included by default).
    """
    raise NotImplementedError("FID requires torchvision.models.inception_v3.")


def compute_inception_score(
    images: torch.Tensor, splits: int = 10,
) -> tuple[float, float]:
    """Inception Score.

    Higher is better. Measures quality and diversity of generated images.

    Requires ``torchvision.models.inception_v3`` (not included by default).

    Returns:
        (mean, std) of IS across splits.
    """
    raise NotImplementedError("IS requires torchvision.models.inception_v3.")
