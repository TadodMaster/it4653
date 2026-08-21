"""Interpolation experiment: demonstrates latent space continuity.

Functions:
    run_vae_interpolation : Generate interpolation grid for a trained VAE
    compare_interpolations: Deprecated (single-VAE version kept for convenience)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

from it4653.data.datasets import get_mnist_loaders
from it4653.models.vae import VAE
from it4653.utils.checkpoints import load_checkpoint

matplotlib.use("Agg")


def run_vae_interpolation(
    checkpoint_path: str,
    dataset: str = "mnist",
    data_root: str = "./data",
    num_pairs: int = 5,
    num_steps: int = 10,
    device: str = "cuda",
    output_dir: str = "./outputs/plots",
) -> str:
    """Run interpolation experiment on a trained VAE and save figure.

    Selects random image pairs from the test set, encodes each to μ,
    linearly interpolates in latent space, and decodes each step.
    The resulting grid demonstrates smooth latent-space transitions.

    Args:
        checkpoint_path: Path to trained VAE checkpoint (.pt).
        dataset: "mnist" or "fashion-mnist".
        data_root: Dataset root.
        num_pairs: Number of image pairs to interpolate.
        num_steps: Number of interpolation steps (including endpoints).
        device: "cuda" or "cpu".
        output_dir: Where to save the output figure.

    Returns:
        Path to saved figure.
    """
    # Load model
    # Infer latent_dim from config if available, otherwise default 32
    model = VAE(latent_dim=32, image_channels=1, image_size=28)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    # Get test dataset
    _, test_loader = get_mnist_loaders(
        dataset=dataset, data_root=data_root, batch_size=num_pairs * 2,
        num_workers=0,
    )
    x_test, _ = next(iter(test_loader))
    x_test = x_test[: num_pairs * 2].to(device)

    # Encode pairs
    with torch.no_grad():
        mu, _ = model.encode(x_test)
        z_pairs = mu.view(num_pairs, 2, model.latent_dim)

        # Linear interpolation for each pair
        alphas = torch.linspace(0, 1, num_steps, device=device)
        grids = []

        for pair_idx in range(num_pairs):
            z1 = z_pairs[pair_idx, 0]
            z2 = z_pairs[pair_idx, 1]
            z_interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
            recon_interp = model.decode(z_interp)
            grids.append(recon_interp.cpu())

        # Assemble multi-row figure: one row per pair
        fig, axes = plt.subplots(num_pairs, num_steps, figsize=(num_steps * 1.2, num_pairs * 1.3))

        if num_pairs == 1:
            axes = axes.reshape(1, -1)

        for row in range(num_pairs):
            for col in range(num_steps):
                ax = axes[row, col]
                ax.imshow(grids[row][col].squeeze(), cmap="gray")
                ax.axis("off")

                # Label top row with alpha values
                if row == 0:
                    ax.set_title(f"{alphas[col].item():.1f}", fontsize=8)

                # Label first column with "Pair N"
                if col == 0:
                    ax.text(-0.1, 0.5, f"Pair {row + 1}",
                            transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=9)

        fig.suptitle(
            f"VAE Latent Space Interpolation (latent_dim={model.latent_dim}, steps={num_steps})",
            fontsize=12,
        )
        fig.tight_layout()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / "interpolation_grid.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Interpolation] Saved to {save_path}")
    return str(save_path)
