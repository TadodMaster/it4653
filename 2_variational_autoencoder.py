#!/usr/bin/env python3
"""
Convolutional Variational Autoencoder (VAE / β-VAE) Experiments on MNIST / Fashion-MNIST

This script trains a Convolutional VAE with configurable latent dimensions,
reconstruction losses, and β (beta) values for the KL-divergence weight.
It is the probabilistic counterpart to a vanilla autoencoder: instead of
learning a deterministic latent vector z, it learns a Gaussian posterior
q(z|x) ≈ N(μ(x), σ²(x)).

Experiments performed:
  1. Reconstruction quality + latent visualisation (2D scatter, interpolation,
     prior samples, latent manifold walk).
  2. Anomaly detection via negative Evidence Lower Bound (ELBO).

Typical call:
    python cvae.py --dataset MNIST --epochs 20 --latent-dims 2 8 32 --losses bce mse --beta 1.0
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard-library imports
# ──────────────────────────────────────────────────────────────────────────────
import argparse                  # Parsing command-line arguments
import random                    # Python-level RNG (seeded for reproducibility)
from pathlib import Path         # Object-oriented filesystem paths

# ──────────────────────────────────────────────────────────────────────────────
# Third-party imports
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt  # Plotting library
import numpy as np               # Numerical computing (arrays, linear algebra)
import pandas as pd              # Tabular data handling (logs, CSV export)

# PyTorch
import torch
import torch.nn as nn            # Neural-network layers
import torch.nn.functional as F  # Stateless functions (losses, activations)
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms   # Standard vision datasets
from torchvision.utils import save_image       # Grid-image saving utility
from tqdm import tqdm            # Progress-bar wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION: Convolutional Variational Autoencoder (VAE)
# ═══════════════════════════════════════════════════════════════════════════════
# What is a VAE?
# --------------
# A Variational Autoencoder (Kingma & Welling, 2014) is a generative model that
# approximates the intractable posterior p(z|x) with a parametric distribution
# q_φ(z|x) — typically a Gaussian whose mean μ(x) and log-variance logσ²(x)
# are output by a neural network (the "encoder").
#
# During training the ELBO (Evidence Lower Bound) is maximised:
#     ELBO = E_q[ log p_θ(x|z) ]  −  β · D_KL( q_φ(z|x) || p(z) )
#     loss = −ELBO   (we therefore minimise this quantity)
#
#   • Reconstruction term (E_q[log p(x|z)]) — how well the decoder recovers x.
#   • KL term (D_KL) — a regulariser that pushes the posterior toward the prior
#     N(0, I), keeping the latent space smooth and preventing over-fitting.
#
# β-VAE: the scalar β > 1 disentangles the latent factors by increasing the
# pressure on the KL term (Higgins et al., 2017).
#
# Reparameterisation trick
# -----------------------
# Sampling z ~ N(μ, σ²) is not differentiable w.r.t. μ and σ because
# gradients do not flow through a random node.  The trick:
#     z = μ + σ · ε,    where ε ~ N(0, I)
# replaces the non-differentiable sample by a deterministic expression plus
# an external noise variable ε, allowing back-propagation.
#
# ═══════════════════════════════════════════════════════════════════════════════

class ConvVAE(nn.Module):
    """
    A fully-convolutional VAE for 28×28 grayscale images.

    Architecture overview
    -----------------------
      Input  (1, 28, 28)
        → Encoder conv blocks : 1 → 32 → 64 channels, spatial 28 → 14 → 7
        → Flatten → FC        : 64·7·7 = 3136
        → Split into two heads: μ  = fc_mu(h)        (latent_dim)
                               logσ² = fc_logvar(h)   (latent_dim)
        → Reparameterize      : z = μ + σ·ε, ε ~ N(0,I)  (latent_dim)
        → FC → Reshape        : latent_dim → 64·7·7 → (64, 7, 7)
        → Decoder transposed-conv blocks : 64 → 32 → 1, spatial 7 → 14 → 28
      Output (1, 28, 28)
    """

    def __init__(self, latent_dim: int):
        """
        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent Gaussian distribution q(z|x).
        """
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder (convolutional feature extractor) ─────────────────────
        # Same conv backbone as a vanilla AE — extracts spatial hierarchies.
        self.encoder_conv = nn.Sequential(
            # 1 → 32 channels,  28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 → 64 channels,  14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Two separate FC heads: one for the posterior mean, one for log variance.
        # Sharing the conv backbone reduces parameters; splitting at the end lets
        # the network specialise one head on location (μ) and one on scale (logσ²).
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # ── Decoder (deterministic reconstruction p_θ(x|z)) ──────────────────
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            # 64 → 32 channels,  7×7 → 14×14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 → 1 channel,  14×14 → 28×28
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),            # Pixel values in [0, 1], matching ToTensor output
        )

    # ── Forward helpers ──────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map an image batch to the parameters of the approximate posterior q(z|x).

        Parameters
        ----------
        x : torch.Tensor
            Images of shape (B, 1, 28, 28).

        Returns
        -------
        mu : torch.Tensor
            Posterior means, shape (B, latent_dim).
        logvar : torch.Tensor
            Posterior log-variances, shape (B, latent_dim).
            We model logσ² rather than σ² for numerical stability:
            exp(logvar) is always positive, and gradients flow well through exp.
        """
        h = self.encoder_conv(x)     # → (B, 64, 7, 7)
        h = h.flatten(start_dim=1)   # → (B, 64*7*7)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample a latent vector z from N(μ, σ²) using the reparameterisation trick.

        Forward pass during training:
            σ  = exp(0.5 · logvar)
            ε  ~ N(0, I)   (standard normal noise, detached from the graph)
            z  = μ + σ · ε

        During evaluation we bypass randomness entirely and use the posterior mean
        μ.  This gives deterministic reconstructions and cleaner visualisations.

        Parameters
        ----------
        mu, logvar : torch.Tensor
            Both of shape (B, latent_dim).

        Returns
        -------
        z : torch.Tensor
            Sampled latent codes, shape (B, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)   # σ from logσ²
            eps = torch.randn_like(std)     # ε ~ N(0, I), same shape as std
            return mu + eps * std            # Differentiable w.r.t. μ and logvar
        return mu                            # Deterministic μ at test time

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a batch of images from latent codes (the generative decoder).

        Parameters
        ----------
        z : torch.Tensor
            Latent codes, shape (B, latent_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed images, shape (B, 1, 28, 28).
        """
        h = self.decoder_fc(z)                   # → (B, 64*7*7)
        h = h.view(z.size(0), 64, 7, 7)         # → (B, 64, 7, 7)
        return self.decoder_conv(h)              # → (B, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE pass: encode → reparameterize → decode.

        Returns
        -------
        x_hat  : reconstructed images  (B, 1, 28, 28)
        z      : sampled latent codes   (B, latent_dim)
        mu     : posterior means          (B, latent_dim)
        logvar : posterior log-variances (B, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY & HARDWARE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """
    Fix every random-number generator for deterministic experiments.
    This includes Python, NumPy, PyTorch CPU/GPU, and the CuDNN backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """
    Select the best available device in the order:
        NVIDIA CUDA GPU → Apple Metal (MPS) → CPU fallback.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataset(name: str, root: Path, train: bool):
    """
    Download (if needed) and return a torchvision MNIST or Fashion-MNIST dataset.

    Parameters
    ----------
    name : str
        "MNIST" or "FashionMNIST".
    root : Path
        Directory for raw data cache.
    train : bool
        True → 60 000-image training split, False → 10 000-image test split.

    Returns
    -------
    torchvision.datasets.VisionDataset
    """
    # ToTensor() converts a PIL Image (H×W, uint8 range [0,255]) into a
    # torch.FloatTensor (C×H×W, normalised to [0,1]).
    transform = transforms.Compose([transforms.ToTensor()])
    if name == "MNIST":
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if name == "FashionMNIST":
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """
    Wrap a dataset in a PyTorch DataLoader with a per-loader seeded RNG.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    batch_size : int
    shuffle : bool
        True for training shuffles; False for deterministic evaluation order.
    seed : int
        Seeded generator so DataLoader shuffling does not perturb the global RNG.
    num_workers : int
        Number of CPU sub-processes for data pre-fetching.  0 = main process.
    device : torch.device
        If CUDA, pin_memory=True allocates page-locked host memory for faster
        asynchronous CPU→GPU transfers.

    Returns
    -------
    DataLoader
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def limit_batches(loader, max_batches: int | None):
    """
    Wrap a DataLoader so it yields at most `max_batches` batches.
    Handy for quick smoke-tests before committing to a full multi-epoch run.
    """
    for batch_id, batch in enumerate(loader):
        if max_batches is not None and batch_id >= max_batches:
            break
        yield batch


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS (VAE-specific)
# ═══════════════════════════════════════════════════════════════════════════════
# The VAE loss is the NEGATIVE ELBO:
#     loss(x) = −E_q[log p(x|z)] + β · D_KL( q(z|x) || p(z) )
#
# We break it into two components:
#   1. Reconstruction loss (negative log-likelihood, averaged over the batch).
#   2. KL divergence (analytical, closed-form for Gaussian posterior vs. N(0,I)).
# ═══════════════════════════════════════════════════════════════════════════════

def recon_loss(x_hat: torch.Tensor, x: torch.Tensor, loss_name: str) -> torch.Tensor:
    """
    Pixel-wise negative log-likelihood, averaged per image then over the batch.

    Why `reduction="sum"` per image?
    -------------------------------
    Using `reduction="mean"` would divide by the total pixel count (784).
    Summing per image and dividing by batch size makes the loss independent
    of image resolution, allowing fair comparison across datasets.

    Parameters
    ----------
    x_hat, x : torch.Tensor
        Tensors of shape (B, 1, 28, 28) in [0, 1].
    loss_name : {"bce", "mse"}
        "bce" — best when the decoder uses Sigmoid and pixels are Bernoulli.
        "mse" — treats reconstruction as regression.

    Returns
    -------
    torch.Tensor
        Scalar: average reconstruction loss per image in the batch.
    """
    if loss_name == "bce":
        return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    if loss_name == "mse":
        return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    raise ValueError(f"Unsupported reconstruction loss: {loss_name}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Analytical D_KL( q(z|x) || N(0, I) ) for diagonal Gaussian posterior.

    Derivation
    ----------
    For a Gaussian q(z|x) = N(μ, diag(σ²)) and prior p(z) = N(0, I):

      D_KL = -0.5 · Σ_j [ 1 + log(σ_j²) − μ_j² − σ_j² ]

    We substitute σ² = exp(logvar).

    This is a closed-form expression so there is no need to sample from q(z|x)
    to compute it — the network outputs μ and logvar directly.

    Parameters
    ----------
    mu, logvar : torch.Tensor
        Both of shape (B, latent_dim).

    Returns
    -------
    torch.Tensor
        Scalar: average KL divergence per image across the batch.
    """
    # Sum over latent dims first (dim=1), then average over the batch.
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size(0)


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    loss_name: str,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assemble the total β-VAE loss and return the individual components too.

    total_loss = recon_loss(x_hat, x) + beta · kl_divergence(mu, logvar)

    Parameters
    ----------
    x_hat, x : (B, 1, 28, 28) tensors [0, 1]
    mu, logvar : (B, latent_dim) posterior parameters
    loss_name : {"bce", "mse"}
        Which reconstruction loss to use.
    beta : float
        Scalar multiplier for the KL term.
        beta = 1.0  → standard VAE (balanced reconstruction & regularisation).
        beta > 1.0  → β-VAE (stronger push toward disentangled, interpretable latents).
        beta < 1.0  → virtually a deterministic AE (weaker regularisation).

    Returns
    -------
    total : torch.Tensor
        Scalar total loss to back-propagate.
    recon : torch.Tensor
        Scalar reconstruction loss (for logging only).
    kl : torch.Tensor
        Scalar KL divergence (for logging only).
    """
    recon = recon_loss(x_hat, x, loss_name)
    kl = kl_divergence(mu, logvar)
    return recon + beta * kl, recon, kl


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION LOOPS
# ═══════════════════════════════════════════════════════════════════════════════
# Unlike the plain AE, the VAE loop tracks THREE numbers per epoch:
#   • total loss (recon + β·KL)  — what gradients are computed on
#   • reconstruction loss          — how good are the reconstructions?
#   • KL divergence              — how far is the posterior from the prior?
# A healthy training trajectory shows the total going down while recon stays
# low and KL converges to a stable, relatively small value.
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, loss_name: str, beta: float, max_batches=None):
    """
    Train the VAE for one epoch, logging total / reconstruction / KL losses.

    Parameters
    ----------
    model : ConvVAE
    loader : DataLoader
    optimizer : torch.optim.Optimizer
    device : torch.device
    loss_name : str
        "bce" or "mse".
    beta : float
        KL weight.
    max_batches : int | None
        Cap batches for debugging.

    Returns
    -------
    (avg_total, avg_recon, avg_kl) : tuple[float, float, float]
    """
    model.train()
    running_total = 0.0  # Accumulated (total loss × batch size)
    running_recon = 0.0  # Accumulated (recon loss × batch size)
    running_kl = 0.0     # Accumulated (KL × batch size)
    seen = 0

    for x, _ in tqdm(limit_batches(loader, max_batches), desc="train", leave=False):
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        x_hat, _, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar, loss_name, beta)
        loss.backward()
        optimizer.step()
        running_total += loss.item() * x.size(0)
        running_recon += recon.item() * x.size(0)
        running_kl += kl.item() * x.size(0)
        seen += x.size(0)

    seen = max(seen, 1)
    return running_total / seen, running_recon / seen, running_kl / seen


@torch.no_grad()
def evaluate(model, loader, device, loss_name: str, beta: float, max_batches=None):
    """
    Evaluate the VAE on a dataset (test or validation) without back-propagation.
    Returns the same 3-tuple as train_one_epoch but on the evaluation split.
    """
    model.eval()
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    seen = 0
    for x, _ in limit_batches(loader, max_batches):
        x = x.to(device)
        x_hat, _, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar, loss_name, beta)
        running_total += loss.item() * x.size(0)
        running_recon += recon.item() * x.size(0)
        running_kl += kl.item() * x.size(0)
        seen += x.size(0)
    seen = max(seen, 1)
    return running_total / seen, running_recon / seen, running_kl / seen


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# VAE-specific visualisations not present in the vanilla AE:
#   • save_samples_from_prior : sample z ~ N(0,I) and decode (generative capability)
#   • plot_latent_manifold    : walk a fine grid in the 2-D prior to see the
#                               learned data manifold
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_reconstruction_grid(model, loader, device, path: Path, n: int = 16) -> None:
    """
    Save a side-by-side grid of original vs. reconstructed images.

    The file contains pairs interleaved in a single tensor:
        [orig_0, recon_0, orig_1, recon_1, ...]
    A companion file `*_original.png` keeps just the originals for reference.
    """
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    original_path = path.parent / f"{path.stem}_original{path.suffix}"
    save_image(x.cpu(), original_path, nrow=8, padding=2)

    x_hat, _, _, _ = model(x)
    pair_rows = torch.empty((2 * n, 1, 28, 28), device=device)
    pair_rows[0::2] = x
    pair_rows[1::2] = x_hat
    save_image(pair_rows.cpu(), path, nrow=8, padding=2)


@torch.no_grad()
def save_samples_from_prior(model, device, path: Path, n: int = 64) -> None:
    """
    Generate *new* images by sampling latent codes from the prior N(0, I) and
    decoding them.

    Why this is unique to VAEs:
    ----------------------------
    A vanilla AE learns a deterministic encoder / decoder.  Even if you sample
    random z values there is no guarantee the decoder will produce meaningful
    images, because the latent codes it saw during training were not necessarily
    distributed as N(0, I).  A VAE, by contrast, explicitly trains the decoder
    to map from N(0, I) to realistic images, making prior sampling a legitimate
    generative operation.

    Parameters
    ----------
    model : ConvVAE
    device : torch.device
    path : Path
        Destination PNG (8×8 grid of 64 images).
    n : int
        Number of samples.
    """
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)   # z ~ N(0, I)
    samples = model.decode(z)
    save_image(samples.cpu(), path, nrow=8, padding=2)


@torch.no_grad()
def plot_latent_2d(model, loader, device, path: Path, max_points: int = 5000) -> None:
    """
    Scatter-plot the *posterior means* μ of test images in 2-D latent space,
    colour-coded by label.  Using the mean (not a stochastic sample) gives a
    deterministic visualisation every time.

    Interpretation
    --------------
    Well-separated coloured clusters imply the VAE has learned latent
    dimensions that correlate with class identity (even though it is
    trained in an entirely unsupervised manner except for labels in the colour).
    """
    model.eval()
    zs = []
    ys = []
    count = 0
    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        zs.append(mu.cpu().numpy())
        ys.append(y.numpy())
        count += x.size(0)
        if count >= max_points:
            break

    z_all = np.concatenate(zs, axis=0)[:max_points]
    y_all = np.concatenate(ys, axis=0)[:max_points]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, s=7, cmap="tab10", alpha=0.8)
    plt.colorbar(scatter, ticks=list(range(10)))
    plt.title("VAE latent space 2D (posterior means)")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def plot_latent_manifold(model, device, path: Path, grid_size: int = 20, span: float = 3.0) -> None:
    """
    Decode a dense uniform grid over the 2-D prior and save the resulting image
    as a single large mosaic.  This is the classic "latent manifold" visualisation
    that reveals how smooth the generative decoder is.

    Only meaningful when latent_dim == 2.  Each cell shows the decoder output at
    a specific (z1, z2) coordinate; walking across the grid should show continuous
    morphs between different digit styles.

    Parameters
    ----------
    model : ConvVAE
    device : torch.device
    path : Path
        Save path for the mosaic PNG.
    grid_size : int
        Number of points along each axis (total grid = grid_size² images).
    span : float
        The prior range to cover (from −span to +span).  ±3σ covers 99.7% of N(0,I)
        but the clusters may lie slightly outside; adjust visually.
    """
    model.eval()
    lin = torch.linspace(-span, span, grid_size, device=device)
    # Build (x, y) grid by nested loops.  `lin.flip(0)` reverses y so the top row
    # of the plot corresponds to the highest y value (matplotlib convention).
    grid = []
    for yi in lin.flip(0):
        for xi in lin:
            grid.append(torch.stack([xi, yi]))
    z = torch.stack(grid)        # (grid_size², 2)
    decoded = model.decode(z)
    save_image(decoded.cpu(), path, nrow=grid_size, padding=1)


@torch.no_grad()
def save_interpolation(model, dataset, device, path: Path, steps: int = 11) -> None:
    """
    Interpolate in latent space between two images from *different* classes
    and decode at each intermediate point.  Shows the smoothness of the
    learned manifold.

    Unlike the AE, the interpolation happens between posterior MEANS μ (not
    deterministic z), because during evaluation the reparameterisation trick
    bypasses randomness.
    """
    model.eval()
    first_x, first_y = dataset[0]
    second_x = None
    for x, y in dataset:
        if y != first_y:
            second_x = x
            break
    if second_x is None:
        raise RuntimeError("Could not find two samples with different labels for interpolation.")

    xa = first_x.unsqueeze(0).to(device)
    xb = second_x.unsqueeze(0).to(device)
    za, _ = model.encode(xa)        # Use deterministic mean μ
    zb, _ = model.encode(xb)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    z = (1 - alphas) * za + alphas * zb
    decoded = model.decode(z)
    save_image(decoded.cpu(), path, nrow=steps, padding=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER (VAE SWEEP)
# ═══════════════════════════════════════════════════════════════════════════════

def train_vae_for_latent_dim(args, latent_dim: int, loss_name: str, train_loader, test_loader, run_dir: Path, device):
    """
    Train ONE VAE configuration (latent_dim × loss_name × beta, taken from args).

    Returns
    -------
    model : ConvVAE
    rows : list[dict]
        One row per epoch, containing total/recon/KL losses for both train and test.
    """
    model = ConvVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rows = []
    for epoch in range(1, args.epochs + 1):
        train_total, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, loss_name, args.beta, args.max_train_batches
        )
        test_total, test_recon, test_kl = evaluate(
            model, test_loader, device, loss_name, args.beta, args.max_test_batches
        )
        row = {
            "model": "VAE",
            "dataset": args.dataset,
            "loss_name": loss_name,
            "latent_dim": latent_dim,
            "beta": args.beta,
            "epoch": epoch,
            "train_total_loss": train_total,
            "train_recon_loss": train_recon,
            "train_kl": train_kl,
            "test_total_loss": test_total,
            "test_recon_loss": test_recon,
            "test_kl": test_kl,
            "seed": args.seed,
        }
        rows.append(row)
        print(row)

    # ── Save artefacts ──
    torch.save(model.state_dict(), run_dir / f"vae_{loss_name}_latent_{latent_dim}.pt")
    save_reconstruction_grid(model, test_loader, device, run_dir / f"recon_{loss_name}_latent_{latent_dim}.png")
    # Prior sampling: this is a VAE super-power — generate *new* fake images
    save_samples_from_prior(model, device, run_dir / f"samples_{loss_name}_latent_{latent_dim}.png")

    if latent_dim == 2:
        plot_latent_2d(model, test_loader, device, run_dir / f"latent_map_2d_{loss_name}.png")
        plot_latent_manifold(model, device, run_dir / f"latent_manifold_{loss_name}.png")
        save_interpolation(model, test_loader.dataset, device, run_dir / f"interpolation_vae_{loss_name}.png")

    return model, rows


def save_loss_comparison(all_rows: list[dict], run_dir: Path) -> None:
    """
    After all VAE configs have finished training, save a summary CSV and a
    two-row plot per loss function:
      Row 0: reconstruction loss vs. latent_dim
      Row 1: KL divergence vs. latent_dim

    This explicitly shows the reconstruction–regularisation trade-off: larger
    latent_dim gives more capacity (lower recon) but usually also larger KL
    because the posterior must stay close to the prior while encoding more info.
    """
    df = pd.DataFrame(all_rows)
    final_df = df.sort_values("epoch").groupby(["loss_name", "latent_dim"], as_index=False).tail(1)
    final_df = final_df.sort_values(["loss_name", "latent_dim"])
    final_df.to_csv(run_dir / "loss_comparison_summary.csv", index=False)

    loss_names = final_df["loss_name"].unique().tolist()
    # 2 rows × len(loss_names) columns; row 0 = recon, row 1 = KL
    fig, axes = plt.subplots(2, len(loss_names), figsize=(6 * len(loss_names), 8), squeeze=False)

    for col, loss_name in enumerate(loss_names):
        loss_df = final_df[final_df["loss_name"] == loss_name]
        ticks = loss_df["latent_dim"].tolist()

        # Reconstruction subplot
        ax = axes[0][col]
        ax.plot(loss_df["latent_dim"], loss_df["test_recon_loss"], marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.set_xlabel("latent dim")
        ax.set_ylabel(f"test reconstruction {loss_name.upper()}")
        ax.set_title(f"VAE trained with {loss_name.upper()}")
        ax.grid(True, alpha=0.3)

        # KL subplot
        ax = axes[1][col]
        ax.plot(loss_df["latent_dim"], loss_df["test_kl"], marker="o", color="tab:orange")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.set_xlabel("latent dim")
        ax.set_ylabel("test KL divergence")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "loss_comparison_by_latent_dim.png", dpi=160)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY-DETECTION EXPERIMENT (VAE variant)
# ═══════════════════════════════════════════════════════════════════════════════
# Anomaly score = negative ELBO = reconstruction BCE + beta · KL per image.
#
# Why ELBO instead of plain reconstruction error?
# ------------------------------------------------
# The VAE's posterior q(z|x) encodes information about how "surprising" x is.
# If x is an anomaly, the encoder will either:
#   (a) Need a posterior far from the prior  →  large KL penalty, OR
#   (b) Struggle to reconstruct it faithfully  →  large reconstruction error.
# Combining both terms (the full negative ELBO) often yields better anomaly
# ranking than either component alone.
# ═══════════════════════════════════════════════════════════════════════════════

def subset_without_digit(dataset, excluded_digit: int) -> Subset:
    """
    Build a subset containing all samples whose label != excluded_digit.
    The resulting subset is the "normal" data used for training the VAE.
    """
    indices = []
    for idx, (_, y) in enumerate(dataset):
        if int(y) != excluded_digit:
            indices.append(idx)
    return Subset(dataset, indices)


@torch.no_grad()
def anomaly_scores(model, loader, device, beta: float, max_batches=None):
    """
    Compute per-image anomaly scores = negative ELBO = recon_BCE + beta · KL.

    Unlike the plain AE which uses only reconstruction error, the VAE score
    incorporates both the quality of reconstruction AND how far the posterior
    deviates from the prior.  Both axes are informative for anomaly detection.

    Parameters
    ----------
    model : ConvVAE
    loader : DataLoader
    device : torch.device
    beta : float
        KL weight (must match the value used during training).
    max_batches : int | None

    Returns
    -------
    scores : np.ndarray
        1-D array of per-image ELBO values (higher = more anomalous).
    labels : np.ndarray
        1-D array of ground-truth integer labels (0-9).
    """
    model.eval()
    scores = []
    labels = []
    for x, y in limit_batches(loader, max_batches):
        x = x.to(device)
        x_hat, _, mu, logvar = model(x)
        # Per-image pixel-wise BCE sum
        recon = F.binary_cross_entropy(x_hat, x, reduction="none").flatten(1).sum(dim=1)
        # Per-image KL sum (no batch averaging here — we need per-image scores)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        score = recon + beta * kl
        scores.extend(score.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.asarray(scores), np.asarray(labels)


def run_anomaly_experiment(args, train_dataset, test_dataset, run_dir: Path, device):
    """
    One-class anomaly detection using the VAE.

    For each digit in --anomaly-digits:
      1. Remove it from training (keep only the other 9 "normal" digits).
      2. Train a VAE on the normal subset.
      3. Score all test images using the negative ELBO.
      4. Compute AUROC, Average Precision, and recall@95%-normal-threshold.
      5. Plot score distributions and decision boundary.
    """
    anomaly_rows = []
    for excluded_digit in args.anomaly_digits:
        # ── Build normal training set and test loader ──
        normal_train = subset_without_digit(train_dataset, excluded_digit)
        train_loader = make_loader(normal_train, args.batch_size, True,  args.seed, args.num_workers, device)
        test_loader  = make_loader(test_dataset,   args.batch_size, False, args.seed, args.num_workers, device)

        # ── Train the VAE on 9 digits ──
        model = ConvVAE(args.anomaly_latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.anomaly_epochs):
            train_one_epoch(
                model, train_loader, optimizer, device, args.anomaly_loss, args.beta, args.max_train_batches
            )

        # ── Score all test images ──
        scores, labels = anomaly_scores(model, test_loader, device, args.beta, args.max_test_batches)
        is_anomaly = (labels == excluded_digit).astype(np.int32)

        # ── Metrics ──
        roc_auc = roc_auc_score(is_anomaly, scores)
        avg_precision = average_precision_score(is_anomaly, scores)
        # Threshold at 95th percentile of normal scores (operational decision boundary)
        threshold = np.percentile(scores[is_anomaly == 0], 95)
        predicted = scores >= threshold
        recall_at_95 = (predicted[is_anomaly == 1].mean()).item()

        row = {
            "excluded_digit": excluded_digit,
            "train_loss_name": args.anomaly_loss,
            "latent_dim": args.anomaly_latent_dim,
            "beta": args.beta,
            "epochs": args.anomaly_epochs,
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "normal_p95_threshold": threshold,
            "anomaly_recall_at_normal_p95": recall_at_95,
            "seed": args.seed,
        }
        anomaly_rows.append(row)
        print(row)

        # ── Plot score distributions ──
        plt.figure(figsize=(7, 4))
        plt.hist(scores[is_anomaly == 0], bins=60, alpha=0.7, label="normal", density=True)
        plt.hist(scores[is_anomaly == 1], bins=60, alpha=0.7, label=f"anomaly digit {excluded_digit}", density=True)
        plt.axvline(threshold, color="black", linestyle="--", linewidth=1, label="normal p95")
        plt.xlabel("negative ELBO score")
        plt.ylabel("density")
        plt.title(f"VAE anomaly detection, excluded digit {excluded_digit}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"anomaly_digit_{excluded_digit}.png", dpi=160)
        plt.close()

    return anomaly_rows


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """
    Define every command-line flag for the VAE experiment.
    The key VAE-specific addition over the AE script is the ``--beta`` knob
    for controlling the KL-divergence weight (β-VAE).
    """
    parser = argparse.ArgumentParser(description="VAE experiments for MNIST/Fashion-MNIST.")

    # ── Dataset ──
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST",
                        help="Dataset to train on (default: MNIST).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory to cache raw data (default: ./data).")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"),
                        help="Directory for run outputs (default: ./runs).")

    # ── Main sweep hyper-parameters ──
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per (loss, latent_dim) config (default: 10).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size (default: 128).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[2, 8, 32, 128],
                        help="Bottleneck sizes to sweep (default: 2 8 32 128).")
    parser.add_argument("--losses", choices=["bce", "mse"], nargs="+", default=["bce", "mse"],
                        help="Reconstruction losses to compare (default: bce mse).")

    # ── β-VAE specific ──
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight on the KL term (default: 1.0).  "
                             "β = 1 → standard VAE.  β > 1 → stronger regularisation.  "
                             "β < 1 → more emphasis on reconstruction.")

    # ── Infrastructure ──
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42).")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker sub-processes (default: 2).")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Cap training batches per epoch (None = all).")
    parser.add_argument("--max-test-batches", type=int, default=None,
                        help="Cap test batches during evaluation (None = all).")

    # ── Anomaly-detection sub-experiment ──
    parser.add_argument("--skip-anomaly", action="store_true",
                        help="Skip the anomaly-detection experiment.")
    parser.add_argument("--anomaly-digits", type=int, nargs="+", default=list(range(10)),
                        help="Digits to treat as anomalies, one-at-a-time (default: 0-9).")
    parser.add_argument("--anomaly-loss", choices=["bce", "mse"], default="bce",
                        help="Loss for the anomaly VAE (default: bce).")
    parser.add_argument("--anomaly-latent-dim", type=int, default=32,
                        help="Bottleneck size for anomaly VAEs (default: 32).")
    parser.add_argument("--anomaly-epochs", type=int, default=5,
                        help="Training epochs for each anomaly VAE (default: 5).")

    return parser.parse_args()


def main():
    """
    Top-level execution flow:
      1. Parse CLI and set seeds / device.
      2. Create run directory that includes beta in the path for easy organisation.
      3. Load data, build loaders.
      4. Sweep over (loss, latent_dim), train, and log every configuration.
      5. Save aggregated CSV and comparison plot (recon + KL).
      6. Optionally run one-class anomaly detection.
    """
    # 1. Parse args
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Unique run directory includes beta so different β runs do not collide
    run_dir = args.out_dir / f"vae_{args.dataset.lower()}_beta{args.beta}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Data
    train_dataset = get_dataset(args.dataset, args.data_dir, train=True)
    test_dataset  = get_dataset(args.dataset, args.data_dir, train=False)
    train_loader = make_loader(train_dataset, args.batch_size, True,  args.seed, args.num_workers, device)
    test_loader  = make_loader(test_dataset,  args.batch_size, False, args.seed, args.num_workers, device)

    # 4. Main sweep
    all_rows = []
    for loss_name in args.losses:
        for latent_dim in args.latent_dims:
            _, rows = train_vae_for_latent_dim(args, latent_dim, loss_name, train_loader, test_loader, run_dir, device)
            all_rows.extend(rows)

    # 5. Save logs and comparison plot
    pd.DataFrame(all_rows).to_csv(run_dir / "experiment_log.csv", index=False)
    save_loss_comparison(all_rows, run_dir)

    # 6. Anomaly detection (optional)
    if not args.skip_anomaly:
        anomaly_rows = run_anomaly_experiment(args, train_dataset, test_dataset, run_dir, device)
        pd.DataFrame(anomaly_rows).to_csv(run_dir / "anomaly_log.csv", index=False)

    print(f"Done. Results saved to: {run_dir.resolve()}")


# Only execute main() when called directly, not on module import.
if __name__ == "__main__":
    main()
