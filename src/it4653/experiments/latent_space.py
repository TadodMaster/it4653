"""Latent-space dimension sweep experiment.

Trains VAEs with different latent_dim values on the same dataset,
measures reconstruction quality (MSE/BCE), and plots:
1. Reconstruction error vs latent_dim
2. Latent-space scatter for the 2D model
3. Number of "active" latent units per dimension

Functions:
    sweep_latent_dims     : Train and evaluate VAEs across latent dimensions
    plot_sweep_results    : Visualize reconstruction error and active units
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

from it4653.data.datasets import get_mnist_loaders
from it4653.evaluation.metrics import (
    compute_active_units,
    compute_reconstruction_error,
    compute_vae_elbo,
)
from it4653.models.vae import VAE
from it4653.training.trainers import train_vae
from it4653.utils.checkpoints import load_checkpoint, save_checkpoint
from it4653.utils.visualization import plot_latent_space_2d

matplotlib.use("Agg")


ResultsDict = dict[str, dict[str, float | list[float]]]


def sweep_latent_dims(
    latent_dims: list[int],
    dataset: str = "mnist",
    data_root: str = "./data",
    batch_size: int = 128,
    image_size: int = 28,
    num_epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cuda",
    output_dir: str = "./outputs/sweep",
) -> ResultsDict:
    """Train VAEs with different latent_dim and collect evaluation metrics.

    For each latent_dim:
    1. Train a VAE from scratch (or reuse checkpoint if present)
    2. Compute test-set reconstruction error (BCE, MSE, MAE)
    3. Compute ELBO components (loss, recon, KL)
    4. Count active latent units

    Args:
        latent_dims: List of latent dimensions to sweep (e.g. [2, 8, 32, 128]).
        dataset: "mnist" or "fashion-mnist".
        data_root: Dataset root.
        batch_size: Batch size.
        image_size: Image spatial size.
        num_epochs: Training epochs per model.
        lr: Adam learning rate.
        beta: KL divergence weight.
        device: "cuda" or "cpu".
        output_dir: Output directory for checkpoints and plots.

    Returns:
        Dict mapping latent_dim → evaluation metrics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared data loaders
    train_loader, test_loader = get_mnist_loaders(
        dataset=dataset, data_root=data_root, batch_size=batch_size,
        image_size=image_size, num_workers=0,
    )

    results: ResultsDict = {}

    for latent_dim in latent_dims:
        print(f"\n{'='*60}")
        print(f"  Latent dim = {latent_dim}")
        print(f"{'='*60}")

        ckpt_path = output_dir / f"vae_dim{latent_dim}.pt"

        # Build model
        model = VAE(latent_dim=latent_dim, image_channels=1, image_size=image_size)

        if ckpt_path.exists():
            print(f"[Resume] Found checkpoint: {ckpt_path}")
            load_checkpoint(str(ckpt_path), model, device=device)
        else:
            # Train from scratch
            history = train_vae(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                num_epochs=num_epochs,
                lr=lr,
                beta=beta,
                device=device,
                save_dir=str(output_dir / "checkpoints"),
                log_dir=str(output_dir / f"logs_dim{latent_dim}"),
                save_every=num_epochs,  # only save final
            )

            # Rename final checkpoint
            final_ckpt = output_dir / "checkpoints" / "vae.pt"
            if final_ckpt.exists():
                os.replace(str(final_ckpt), str(ckpt_path))

        # Evaluation
        print(f"  [Eval] Computing metrics...")
        recon = compute_reconstruction_error(model, test_loader, device=device)
        elbo = compute_vae_elbo(model, test_loader, beta=beta, device=device)
        active = compute_active_units(model, test_loader, device=device)

        results[latent_dim] = {
            "recon_loss": float(recon["recon_loss"]),
            "mse": float(recon["mse"]),
            "mae": float(recon["mae"]),
            "elbo_loss": float(elbo["loss"]),
            "elbo_recon": float(elbo["recon"]),
            "elbo_kl": float(elbo["kl"]),
            "active_units": active["active"],
            "total_units": active["total"],
        }

        print(f"  Results: recon={recon['recon_loss']:.4f}, mse={recon['mse']:.4f}, "
              f"elbo={elbo['loss']:.4f}, active_units={active['active']}/{active['total']}")

        # Special save: 2D latent space scatter plot
        if latent_dim == 2:
            plot_path = output_dir / "latent_space_2d.png"
            plot_latent_space_2d(
                model, test_loader, device=device, num_samples=5000,
                save_path=str(plot_path),
            )
            print(f"  [Plot] Saved latent space 2D: {plot_path}")

    # Save JSON results
    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Sweep] Results saved to {json_path}")

    return results


def plot_sweep_results(
    results: ResultsDict,
    save_path: str = "./outputs/sweep/sweep_comparison.png",
) -> plt.Figure:
    """Plot reconstruction error and active units vs latent_dim.

    Args:
        results: Output from sweep_latent_dims().
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure handle.
    """
    dims = sorted(results.keys())
    recon_losses = [results[d]["recon_loss"] for d in dims]
    mses = [results[d]["mse"] for d in dims]
    elbo_losses = [results[d]["elbo_loss"] for d in dims]
    active = [results[d]["active_units"] for d in dims]
    total = [results[d]["total_units"] for d in dims]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reconstruction error (BCE)
    axes[0, 0].plot(dims, recon_losses, marker="o", color="steelblue")
    axes[0, 0].set_xlabel("Latent Dimension")
    axes[0, 0].set_ylabel("BCE Reconstruction Error")
    axes[0, 0].set_title("Reconstruction Error vs Latent Dim")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(dims)

    # MSE
    axes[0, 1].plot(dims, mses, marker="o", color="forestgreen")
    axes[0, 1].set_xlabel("Latent Dimension")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("MSE vs Latent Dim")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(dims)

    # ELBO
    axes[1, 0].plot(dims, elbo_losses, marker="o", color="crimson")
    axes[1, 0].set_xlabel("Latent Dimension")
    axes[1, 0].set_ylabel("ELBO Loss")
    axes[1, 0].set_title("ELBO vs Latent Dim")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(dims)

    # Active units
    axes[1, 1].bar(range(len(dims)), active, color="goldenrod", label="active")
    axes[1, 1].bar(range(len(dims)), [t - a for t, a in zip(total, active)],
                   bottom=active, color="lightgray", label="inactive")
    axes[1, 1].set_xticks(range(len(dims)))
    axes[1, 1].set_xticklabels(dims)
    axes[1, 1].set_xlabel("Latent Dimension")
    axes[1, 1].set_ylabel("Units")
    axes[1, 1].set_title("Active vs Inactive Latent Units")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Sweep comparison saved to {save_path}")

    return fig
