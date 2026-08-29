#!/usr/bin/env python3
"""
Conditional Convolutional Variational Autoencoder (CVAE) Experiments on MNIST / Fashion-MNIST

This script is the *conditional* counterpart of ``cvae.py``.  A plain VAE models
p(x); a Conditional VAE (Sohn et al., 2015) models p(x | y), where y is a class
label.  Both the encoder and the decoder receive the label, so:

  • the latent code z no longer has to store *which* digit is drawn — that
    information is handed to the decoder for free — and is therefore pushed
    toward encoding only the *style* (stroke width, slant, thickness);
  • generation becomes controllable: pick the class you want, sample z ~ N(0, I),
    and the decoder produces an image of exactly that class.

Experiments performed:
  1. Reconstruction quality + latent visualisation (2-D scatter, per-class latent
     manifold, z / label interpolation).
  2. Controllable generation: class-conditional samples and label-swap
     ("same style, different digit") grids.
  3. Anomaly detection via the conditional negative ELBO, with two scoring modes
     (see ``--anomaly-score-mode``).

Related scripts:
  • cae.py / Autoencoder_final.py — deterministic convolutional autoencoder
  • cvae.py                       — unconditional VAE / β-VAE

Typical call:
    python conditional_vae.py --dataset MNIST --epochs 20 --latent-dims 2 8 32 --losses bce mse --beta 1.0
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
import torch.nn.functional as F  # Stateless functions (losses, activations, one_hot)
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms   # Standard vision datasets
from torchvision.utils import save_image       # Grid-image saving utility
from tqdm import tqdm            # Progress-bar wrapper


# Both MNIST and Fashion-MNIST have exactly ten classes, so the conditioning
# vector is a 10-dimensional one-hot code throughout this script.
NUM_CLASSES = 10


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL-CONDITIONING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def one_hot_labels(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Convert a batch of integer class labels into float one-hot vectors.

    Why one-hot rather than the raw integer?
    ---------------------------------------
    Feeding the label as a single scalar would impose a fake ordering on the
    classes ("3 is between 2 and 4"), which is meaningless for digit identity.
    A one-hot code makes every class an orthogonal direction, so the network can
    learn an independent effect per class.  (An nn.Embedding would work too and
    scales better to hundreds of classes; with only ten, one-hot is simplest.)

    Parameters
    ----------
    y : torch.Tensor
        Integer labels of shape (B,), dtype int64.
    num_classes : int
        Size of the one-hot vector (10 here).
    device : torch.device
        Device the result should live on.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (B, num_classes) with a single 1.0 per row.
    """
    return F.one_hot(y.to(device), num_classes=num_classes).float()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION: Conditional Convolutional VAE (CVAE)
# ═══════════════════════════════════════════════════════════════════════════════
# What changes relative to the plain VAE?
# ---------------------------------------
# The VAE maximises the ELBO of the marginal likelihood p(x); the CVAE maximises
# the ELBO of the *conditional* likelihood p(x | y):
#
#     log p(x|y) ≥ E_q[ log p_θ(x | z, y) ] − β · D_KL( q_φ(z | x, y) || p(z|y) )
#     loss = −ELBO   (minimised)
#
# We use the common simplification p(z|y) = p(z) = N(0, I): the prior is kept
# class-independent so that a single N(0, I) sample is valid for every class and
# the KL term stays the same closed-form expression as in the plain VAE.
#
# The label enters the network in two places:
#   1. Encoder q(z | x, y): the one-hot label is broadcast into `num_classes`
#      constant 28×28 channels and concatenated to the image, so the first Conv2d
#      takes 1 + 10 = 11 input channels.  Spatial broadcasting is the standard
#      trick for injecting a vector into a convolutional stack.
#   2. Decoder p(x | z, y): the one-hot label is concatenated to z before the
#      first fully-connected layer, so it takes latent_dim + 10 inputs.
#
# Consequence — the "information short-cut":
# Because the decoder is *told* the class, storing class identity inside z buys
# no reduction in reconstruction error but still costs KL.  The optimum therefore
# strips class information out of z, leaving z to carry style.  This is exactly
# what makes label-swapping (same z, different y) work.
# ═══════════════════════════════════════════════════════════════════════════════

class ConvCVAE(nn.Module):
    """
    A fully-convolutional Conditional VAE for 28×28 grayscale images.

    Architecture overview
    ---------------------
      Input  x (1, 28, 28)  +  label y (one-hot, 10)
        → Broadcast y to (10, 28, 28) and concatenate → (11, 28, 28)
        → Encoder conv blocks : 11 → 32 → 64 channels, spatial 28 → 14 → 7
        → Flatten → FC        : 64·7·7 = 3136
        → Split into two heads: μ     = fc_mu(h)      (latent_dim)
                                logσ² = fc_logvar(h)  (latent_dim)
        → Reparameterize      : z = μ + σ·ε, ε ~ N(0,I)   (latent_dim)
        → Concatenate y       : [z, y]  (latent_dim + 10)
        → FC → Reshape        : latent_dim+10 → 64·7·7 → (64, 7, 7)
        → Decoder transposed-conv blocks : 64 → 32 → 1, spatial 7 → 14 → 28
      Output (1, 28, 28)
    """

    def __init__(self, latent_dim: int, num_classes: int = NUM_CLASSES):
        """
        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent Gaussian q(z | x, y).  A CVAE can afford
            a smaller latent than a comparable VAE because class identity is
            supplied externally instead of being encoded in z.
        num_classes : int
            Number of conditioning classes (10 for MNIST / Fashion-MNIST).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ── Encoder q(z | x, y) ───────────────────────────────────────────────
        # Identical to the VAE backbone except for the input channel count:
        # 1 image channel + `num_classes` broadcast label channels.
        self.encoder_conv = nn.Sequential(
            # 1+10 → 32 channels,  28×28 → 14×14
            nn.Conv2d(1 + num_classes, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 → 64 channels,  14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Two FC heads sharing one conv backbone: posterior mean and log-variance.
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # ── Decoder p(x | z, y) ───────────────────────────────────────────────
        # The label is concatenated to the latent vector, so the input width of
        # the first FC layer grows by num_classes.
        self.decoder_fc = nn.Linear(latent_dim + num_classes, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            # 64 → 32 channels,  7×7 → 14×14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 → 1 channel,  14×14 → 28×28
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),            # Pixel values in [0, 1], matching ToTensor output
        )

    # ── Forward helpers ───────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, y_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map an image batch *and its labels* to the parameters of q(z | x, y).

        Parameters
        ----------
        x : torch.Tensor
            Images of shape (B, 1, 28, 28).
        y_onehot : torch.Tensor
            Float one-hot labels of shape (B, num_classes).

        Returns
        -------
        mu : torch.Tensor
            Posterior means, shape (B, latent_dim).
        logvar : torch.Tensor
            Posterior log-variances, shape (B, latent_dim).  We model logσ² rather
            than σ² for numerical stability: exp(logvar) is positive by
            construction and gradients flow well through exp.
        """
        # Turn the (B, 10) one-hot vector into (B, 10, 28, 28) constant maps, so
        # every spatial position of the conv stack can "see" the class.
        # expand() creates a broadcast view — no memory is actually copied until
        # the following cat().
        y_maps = y_onehot.view(y_onehot.size(0), self.num_classes, 1, 1)
        y_maps = y_maps.expand(-1, -1, x.size(2), x.size(3))
        xy = torch.cat([x, y_maps], dim=1)   # → (B, 1+num_classes, 28, 28)

        h = self.encoder_conv(xy)    # → (B, 64, 7, 7)
        h = h.flatten(start_dim=1)   # → (B, 64*7*7)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample z from N(μ, σ²) with the reparameterisation trick.

        Sampling is not differentiable w.r.t. μ and σ because gradients cannot
        flow through a random node.  The trick moves the randomness into an
        external variable ε:
            σ = exp(0.5 · logvar)
            ε ~ N(0, I)
            z = μ + σ · ε
        which *is* differentiable w.r.t. μ and logvar.

        At evaluation time we drop the noise and return μ, giving deterministic
        reconstructions and reproducible visualisations.

        Parameters
        ----------
        mu, logvar : torch.Tensor
            Both of shape (B, latent_dim).

        Returns
        -------
        z : torch.Tensor
            Latent codes, shape (B, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)   # σ from logσ²
            eps = torch.randn_like(std)     # ε ~ N(0, I)
            return mu + eps * std           # Differentiable w.r.t. μ and logvar
        return mu                           # Deterministic μ at test time

    def decode(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        """
        Generate images from latent codes *and* target labels: p(x | z, y).

        This is the controllable-generation entry point — change `y_onehot` while
        holding z fixed and the same style is redrawn as a different class.

        Parameters
        ----------
        z : torch.Tensor
            Latent codes, shape (B, latent_dim).
        y_onehot : torch.Tensor
            Float one-hot labels, shape (B, num_classes).  Soft (non-one-hot)
            vectors are also accepted, which is what makes label interpolation
            possible.

        Returns
        -------
        x_hat : torch.Tensor
            Generated images, shape (B, 1, 28, 28).
        """
        zy = torch.cat([z, y_onehot], dim=1)     # → (B, latent_dim + num_classes)
        h = self.decoder_fc(zy)                  # → (B, 64*7*7)
        h = h.view(z.size(0), 64, 7, 7)          # → (B, 64, 7, 7)
        return self.decoder_conv(h)              # → (B, 1, 28, 28)

    def forward(
        self, x: torch.Tensor, y_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full CVAE pass: encode(x, y) → reparameterize → decode(z, y).

        Note the same label is used on both sides during training — the model is
        being taught to reconstruct x *given* that its class is y.

        Returns
        -------
        x_hat  : reconstructed images     (B, 1, 28, 28)
        z      : sampled latent codes     (B, latent_dim)
        mu     : posterior means          (B, latent_dim)
        logvar : posterior log-variances  (B, latent_dim)
        """
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y_onehot)
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
# Identical to the VAE script — except that the labels, which the plain VAE threw
# away with `for x, _ in loader`, are now an essential model input.
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
# LOSS FUNCTIONS (identical in form to the plain VAE)
# ═══════════════════════════════════════════════════════════════════════════════
# The conditional negative ELBO is:
#     loss(x, y) = −E_q[log p(x | z, y)] + β · D_KL( q(z | x, y) || N(0, I) )
#
# Conditioning changes *what the networks see*, not the algebra of the objective:
# the reconstruction term is still a pixel-wise negative log-likelihood, and with
# a class-independent prior N(0, I) the KL term keeps its closed form.
# ═══════════════════════════════════════════════════════════════════════════════

def recon_loss(x_hat: torch.Tensor, x: torch.Tensor, loss_name: str) -> torch.Tensor:
    """
    Pixel-wise negative log-likelihood, summed per image then averaged over the batch.

    Why `reduction="sum"` per image?
    -------------------------------
    Using `reduction="mean"` would divide by the total pixel count (784), tying
    the reported number to the image resolution.  Summing per image and dividing
    by the batch size keeps the scale resolution-independent, so CVAE numbers stay
    directly comparable with the AE / VAE scripts.

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
    Analytical D_KL( q(z | x, y) || N(0, I) ) for a diagonal Gaussian posterior.

    Derivation
    ----------
    For q = N(μ, diag(σ²)) and prior p(z) = N(0, I):

      D_KL = −0.5 · Σ_j [ 1 + log(σ_j²) − μ_j² − σ_j² ]

    substituting σ² = exp(logvar).  Closed form, so no sampling is needed here —
    the encoder hands us μ and logvar directly.

    Expected CVAE behaviour: this term usually settles *lower* than in an
    unconditional VAE of the same latent size, because the posterior no longer
    needs to move far from the prior to encode class identity.

    Parameters
    ----------
    mu, logvar : torch.Tensor
        Both of shape (B, latent_dim).

    Returns
    -------
    torch.Tensor
        Scalar: average KL divergence per image across the batch.
    """
    # Sum over latent dims, then average over the batch.
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size(0)


def cvae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    loss_name: str,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assemble the total β-weighted conditional VAE loss and return its components.

    total_loss = recon_loss(x_hat, x) + beta · kl_divergence(mu, logvar)

    Parameters
    ----------
    x_hat, x : (B, 1, 28, 28) tensors in [0, 1]
    mu, logvar : (B, latent_dim) posterior parameters
    loss_name : {"bce", "mse"}
        Which reconstruction loss to use.
    beta : float
        Multiplier on the KL term.
        beta = 1.0 → standard CVAE.
        beta > 1.0 → stronger regularisation; z is pushed even harder toward pure
                     style, since any class information it retains costs KL.
        beta < 1.0 → closer to a deterministic conditional autoencoder.

    Returns
    -------
    total : torch.Tensor
        Scalar total loss to back-propagate.
    recon : torch.Tensor
        Scalar reconstruction loss (logging only).
    kl : torch.Tensor
        Scalar KL divergence (logging only).
    """
    recon = recon_loss(x_hat, x, loss_name)
    kl = kl_divergence(mu, logvar)
    return recon + beta * kl, recon, kl


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION LOOPS
# ═══════════════════════════════════════════════════════════════════════════════
# Like the VAE loop these track three numbers per epoch (total / recon / KL).
# The only structural difference: the label `y` returned by the DataLoader is no
# longer discarded — it is one-hot encoded and fed to both encoder and decoder.
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, loss_name: str, beta: float, max_batches=None):
    """
    Train the CVAE for one epoch, logging total / reconstruction / KL losses.

    Parameters
    ----------
    model : ConvCVAE
    loader : DataLoader
        Must yield (image, label) pairs — labels are required here.
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

    for x, y in tqdm(limit_batches(loader, max_batches), desc="train", leave=False):
        x = x.to(device)
        y_onehot = one_hot_labels(y, model.num_classes, device)   # Conditioning signal
        optimizer.zero_grad(set_to_none=True)
        x_hat, _, mu, logvar = model(x, y_onehot)
        loss, recon, kl = cvae_loss(x_hat, x, mu, logvar, loss_name, beta)
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
    Evaluate the CVAE on a split without back-propagation.
    Returns the same 3-tuple as train_one_epoch.

    Because `model.eval()` makes reparameterize() return μ, the evaluation numbers
    are deterministic given the data order.
    """
    model.eval()
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    seen = 0
    for x, y in limit_batches(loader, max_batches):
        x = x.to(device)
        y_onehot = one_hot_labels(y, model.num_classes, device)
        x_hat, _, mu, logvar = model(x, y_onehot)
        loss, recon, kl = cvae_loss(x_hat, x, mu, logvar, loss_name, beta)
        running_total += loss.item() * x.size(0)
        running_recon += recon.item() * x.size(0)
        running_kl += kl.item() * x.size(0)
        seen += x.size(0)
    seen = max(seen, 1)
    return running_total / seen, running_recon / seen, running_kl / seen


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# CVAE-specific visualisations that have no unconditional equivalent:
#   • save_class_conditional_samples : one row per class, columns share the same z
#                                      → controllable generation + style transfer
#   • save_label_swap_grid           : encode a real image, redraw it as all
#                                      ten classes → proves z holds style, not class
#   • save_label_interpolation       : morph the *label* vector while z is fixed
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_reconstruction_grid(model, loader, device, path: Path, n: int = 16) -> None:
    """
    Save a side-by-side grid of original vs. reconstructed images.

    The file contains pairs interleaved in a single tensor:
        [orig_0, recon_0, orig_1, recon_1, ...]
    A companion file `*_original.png` keeps just the originals for reference.

    Reconstruction is conditioned on each image's *true* label, which is the same
    setting the model was trained under.
    """
    model.eval()
    x, y = next(iter(loader))
    x = x[:n].to(device)
    y_onehot = one_hot_labels(y[:n], model.num_classes, device)

    original_path = path.parent / f"{path.stem}_original{path.suffix}"
    save_image(x.cpu(), original_path, nrow=8, padding=2)

    x_hat, _, _, _ = model(x, y_onehot)
    pair_rows = torch.empty((2 * n, 1, 28, 28), device=device)
    pair_rows[0::2] = x
    pair_rows[1::2] = x_hat
    save_image(pair_rows.cpu(), path, nrow=8, padding=2)


@torch.no_grad()
def save_class_conditional_samples(model, device, path: Path, per_class: int = 8) -> None:
    """
    Generate new images with *chosen* classes: the headline capability of a CVAE.

    Layout
    ------
    Row c  = class c (0-9, top to bottom).
    Column j shares one latent vector z_j across **all** rows.  Reading down a
    column therefore shows the same style rendered as each of the ten classes;
    reading across a row shows style diversity within one class.

    Why an unconditional VAE cannot do this
    ---------------------------------------
    A plain VAE samples z ~ N(0, I) and gets whatever class that region of the
    latent space happens to encode — you cannot ask it for "a 7".  Here the class
    is an explicit input, so generation is controllable, and because the KL term
    discourages z from storing class identity, the same z stays visually
    consistent as the label changes.

    Parameters
    ----------
    model : ConvCVAE
    device : torch.device
    path : Path
        Destination PNG (num_classes rows × per_class columns).
    per_class : int
        Number of samples (columns) per class.
    """
    model.eval()
    num_classes = model.num_classes

    # One shared set of styles, reused for every class.
    z = torch.randn(per_class, model.latent_dim, device=device)

    tiles = []
    for c in range(num_classes):
        labels = torch.full((per_class,), c, dtype=torch.long, device=device)
        y_onehot = one_hot_labels(labels, num_classes, device)
        tiles.append(model.decode(z, y_onehot))       # (per_class, 1, 28, 28)

    samples = torch.cat(tiles, dim=0)                 # (num_classes*per_class, 1, 28, 28)
    # nrow=per_class ⇒ each class occupies exactly one row of the grid.
    save_image(samples.cpu(), path, nrow=per_class, padding=2)


@torch.no_grad()
def save_label_swap_grid(model, loader, device, path: Path, n: int = 8) -> None:
    """
    Take real test images, encode them, then re-decode with every possible label.

    Layout
    ------
    Each row is: [ original image | decoded as class 0 | ... | decoded as class 9 ]
    so the grid is n rows × (1 + num_classes) columns.

    What it demonstrates
    --------------------
    The latent code z is held fixed across a row while only the conditioning label
    changes.  If the CVAE has learned the intended split, every cell in a row keeps
    the *style* of the original (slant, stroke weight) while showing the *identity*
    demanded by its column.  Leakage of class information into z shows up here as
    rows that refuse to change identity, or as distortions in the off-diagonal
    cells.

    Note the encoder is given each image's true label — that is the setting it was
    trained on, so μ is the style code the decoder expects.
    """
    model.eval()
    num_classes = model.num_classes

    x, y = next(iter(loader))
    x = x[:n].to(device)
    y_true = one_hot_labels(y[:n], num_classes, device)

    # Style codes: posterior means of the real images (eval mode ⇒ no sampling).
    mu, _ = model.encode(x, y_true)                   # (n, latent_dim)

    rows = []
    for i in range(x.size(0)):
        # Column 0: the untouched original, for visual reference.
        row = [x[i:i + 1]]
        # Repeat this single style code once per target class.
        z_i = mu[i:i + 1].expand(num_classes, -1)     # (num_classes, latent_dim)
        all_labels = torch.arange(num_classes, device=device)
        row.append(model.decode(z_i, one_hot_labels(all_labels, num_classes, device)))
        rows.append(torch.cat(row, dim=0))            # (1 + num_classes, 1, 28, 28)

    grid = torch.cat(rows, dim=0)
    save_image(grid.cpu(), path, nrow=1 + num_classes, padding=2)


@torch.no_grad()
def plot_latent_2d(model, loader, device, path: Path, max_points: int = 5000) -> None:
    """
    Scatter-plot the posterior means μ of test images in 2-D latent space,
    coloured by the true label.

    Interpretation — and it is the OPPOSITE of the VAE case
    -------------------------------------------------------
    For an unconditional VAE, well-separated colour clusters are the sign of a
    good latent space.  For a CVAE they are a warning: the decoder already knows
    the class, so any class separation still visible in z means class information
    is leaking into the latent code instead of being left to the label.  The ideal
    CVAE plot is a single blended cloud roughly matching the N(0, I) prior, with
    the colours thoroughly mixed.
    """
    model.eval()
    zs = []
    ys = []
    count = 0
    for x, y in loader:
        x = x.to(device)
        y_onehot = one_hot_labels(y, model.num_classes, device)
        mu, _ = model.encode(x, y_onehot)
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
    plt.title("CVAE latent space 2D (posterior means)\nmixed colours = class info left to the label")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def plot_latent_manifold(
    model, device, path: Path, class_label: int, grid_size: int = 20, span: float = 3.0
) -> None:
    """
    Decode a dense uniform grid over the 2-D prior *for one fixed class* and save
    the result as a single mosaic.

    Only meaningful when latent_dim == 2.  Unlike the VAE manifold — where moving
    across the grid morphs between different digits — every cell here shows the
    same class, so the axes reveal the pure *style* factors the model has learned
    (slant, thickness, width).

    Parameters
    ----------
    model : ConvCVAE
    device : torch.device
    path : Path
        Save path for the mosaic PNG.
    class_label : int
        The class held fixed across the whole grid.
    grid_size : int
        Points per axis (total = grid_size² images).
    span : float
        Prior range covered, from −span to +span.  ±3σ covers 99.7 % of N(0, I).
    """
    model.eval()
    lin = torch.linspace(-span, span, grid_size, device=device)
    # `lin.flip(0)` reverses the y axis so the top row of the mosaic corresponds to
    # the largest y value, matching the usual plotting convention.
    grid = []
    for yi in lin.flip(0):
        for xi in lin:
            grid.append(torch.stack([xi, yi]))
    z = torch.stack(grid)                                  # (grid_size², 2)

    labels = torch.full((z.size(0),), class_label, dtype=torch.long, device=device)
    decoded = model.decode(z, one_hot_labels(labels, model.num_classes, device))
    save_image(decoded.cpu(), path, nrow=grid_size, padding=1)


@torch.no_grad()
def save_interpolation(model, dataset, device, path: Path, steps: int = 11) -> None:
    """
    Two-row latent interpolation between two test images of *different* classes.

    Row 0 — style-only walk: interpolate z from image A to image B while the label
            stays pinned to class A.  Identity is constant; only style morphs.
    Row 1 — joint walk: interpolate z *and* the one-hot label together, so the
            image gradually turns into the other class as well.  This is possible
            because decode() accepts soft label vectors, not just one-hot ones.

    Comparing the two rows isolates the two axes of variation the CVAE separates:
    what the latent controls versus what the condition controls.
    """
    model.eval()
    num_classes = model.num_classes

    # Endpoint A: the first sample in the dataset.
    first_x, first_y = dataset[0]
    # Endpoint B: scan forward for the first sample of a DIFFERENT class.
    second_x = None
    second_y = None
    for x, y in dataset:
        if y != first_y:
            second_x = x
            second_y = y
            break
    if second_x is None:
        raise RuntimeError("Could not find two samples with different labels for interpolation.")

    xa = first_x.unsqueeze(0).to(device)
    xb = second_x.unsqueeze(0).to(device)
    ya = one_hot_labels(torch.tensor([int(first_y)]), num_classes, device)    # (1, num_classes)
    yb = one_hot_labels(torch.tensor([int(second_y)]), num_classes, device)

    # Style codes: posterior means, each encoded under its own true label.
    za, _ = model.encode(xa, ya)
    zb, _ = model.encode(xb, yb)

    # α ∈ [0, 1], shaped (steps, 1) so it broadcasts over the latent / label dims.
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    z_interp = (1 - alphas) * za + alphas * zb                 # (steps, latent_dim)

    # Row 0: label fixed at class A.
    row_fixed_label = model.decode(z_interp, ya.expand(steps, -1))
    # Row 1: label morphs alongside z (soft one-hot in between).
    y_interp = (1 - alphas) * ya + alphas * yb                 # (steps, num_classes)
    row_joint = model.decode(z_interp, y_interp)

    decoded = torch.cat([row_fixed_label, row_joint], dim=0)   # (2*steps, 1, 28, 28)
    save_image(decoded.cpu(), path, nrow=steps, padding=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER (CVAE SWEEP)
# ═══════════════════════════════════════════════════════════════════════════════

def train_cvae_for_latent_dim(args, latent_dim: int, loss_name: str, train_loader, test_loader, run_dir: Path, device):
    """
    Train ONE CVAE configuration (latent_dim × loss_name × beta, taken from args).

    Returns
    -------
    model : ConvCVAE
    rows : list[dict]
        One row per epoch, containing total/recon/KL losses for train and test.
    """
    model = ConvCVAE(latent_dim, num_classes=NUM_CLASSES).to(device)
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
            "model": "CVAE",
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
    torch.save(model.state_dict(), run_dir / f"cvae_{loss_name}_latent_{latent_dim}.pt")
    save_reconstruction_grid(model, test_loader, device, run_dir / f"recon_{loss_name}_latent_{latent_dim}.png")
    # Controllable generation: one row per class, columns sharing a latent style.
    save_class_conditional_samples(
        model, device, run_dir / f"samples_by_class_{loss_name}_latent_{latent_dim}.png",
        per_class=args.samples_per_class,
    )
    # Style/identity split: real images redrawn as every other class.
    save_label_swap_grid(model, test_loader, device, run_dir / f"label_swap_{loss_name}_latent_{latent_dim}.png")

    # 2-D-only visualisations: the scatter and the per-class manifolds are only
    # interpretable when the latent space really has two axes.
    if latent_dim == 2:
        plot_latent_2d(model, test_loader, device, run_dir / f"latent_map_2d_{loss_name}.png")
        save_interpolation(model, test_loader.dataset, device, run_dir / f"interpolation_cvae_{loss_name}.png")
        # One style manifold per class — each shows what z controls for that digit.
        for class_label in range(NUM_CLASSES):
            plot_latent_manifold(
                model, device,
                run_dir / f"latent_manifold_{loss_name}_class{class_label}.png",
                class_label=class_label,
            )

    return model, rows


def save_loss_comparison(all_rows: list[dict], run_dir: Path) -> None:
    """
    After all CVAE configs have trained, save a summary CSV and a two-row plot per
    loss function:
      Row 0: reconstruction loss vs. latent_dim
      Row 1: KL divergence vs. latent_dim

    Read alongside the equivalent plot from ``cvae.py``, this is the clearest
    quantitative evidence of what conditioning buys: at a given latent_dim the
    CVAE typically reaches a lower reconstruction loss *and* a lower KL, because
    the class information it no longer has to squeeze through z is delivered by
    the label instead.
    """
    df = pd.DataFrame(all_rows)
    final_df = df.sort_values("epoch").groupby(["loss_name", "latent_dim"], as_index=False).tail(1)
    final_df = final_df.sort_values(["loss_name", "latent_dim"])
    final_df.to_csv(run_dir / "loss_comparison_summary.csv", index=False)

    loss_names = final_df["loss_name"].unique().tolist()
    # 2 rows × len(loss_names) columns; row 0 = recon, row 1 = KL.
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
        ax.set_title(f"CVAE trained with {loss_name.upper()}")
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
# ANOMALY-DETECTION EXPERIMENT (CVAE variant)
# ═══════════════════════════════════════════════════════════════════════════════
# Anomaly score = conditional negative ELBO = recon_BCE + beta · KL, per image.
#
# Conditioning raises a question the unconditional VAE never faces: which label do
# we condition on at scoring time?  The held-out digit's label was never seen in
# training, so its slot in the one-hot vector is attached to untrained weights.
# Two honest answers, selected with --anomaly-score-mode:
#
#   "min-over-labels" (default, realistic)
#       Score the image under every label the model was actually trained on and
#       keep the minimum: "how well can the best-fitting known class explain this
#       image?"  Requires no label at test time, which is the realistic deployment
#       setting for anomaly detection, and turns the CVAE into an ensemble of nine
#       class-conditional density models.
#
#   "true-label" (diagnostic)
#       Condition on the ground-truth label, including the unseen one.  Cheaper
#       (one forward pass) but partly measures "this label was never trained"
#       rather than "this image looks unusual", so it tends to flatter the metrics.
# ═══════════════════════════════════════════════════════════════════════════════

def subset_without_digit(dataset, excluded_digit: int) -> Subset:
    """
    Build a subset containing all samples whose label != excluded_digit.
    This is the "normal" data the CVAE is trained on; the excluded class is the
    anomaly at test time, and its conditioning slot stays untrained.
    """
    indices = []
    for idx, (_, y) in enumerate(dataset):
        if int(y) != excluded_digit:
            indices.append(idx)
    return Subset(dataset, indices)


@torch.no_grad()
def conditional_neg_elbo(model, x: torch.Tensor, y_onehot: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Per-image conditional negative ELBO = recon_BCE + beta · KL.

    Both terms are summed per image (no batch averaging) because we need one
    score per sample to rank them.

    Parameters
    ----------
    model : ConvCVAE
    x : torch.Tensor
        Images, (B, 1, 28, 28).
    y_onehot : torch.Tensor
        Conditioning labels, (B, num_classes).
    beta : float
        KL weight — should match the value used in training.

    Returns
    -------
    torch.Tensor
        Scores of shape (B,); higher = worse explained = more anomalous.
    """
    x_hat, _, mu, logvar = model(x, y_onehot)
    # Per-image pixel-wise BCE sum.
    recon = F.binary_cross_entropy(x_hat, x, reduction="none").flatten(1).sum(dim=1)
    # Per-image KL sum over the latent dimensions.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon + beta * kl


@torch.no_grad()
def anomaly_scores(model, loader, device, beta: float, mode: str, candidate_labels, max_batches=None):
    """
    Compute one anomaly score per test image using the conditional negative ELBO.

    Parameters
    ----------
    model : ConvCVAE
    loader : DataLoader
        Test loader containing ALL classes (normal + the held-out one).
    device : torch.device
    beta : float
        KL weight (match the training value).
    mode : {"min-over-labels", "true-label"}
        See the section banner above.
    candidate_labels : list[int]
        Labels the model was trained on — the only ones "min-over-labels" tries.
    max_batches : int | None

    Returns
    -------
    scores : np.ndarray
        1-D array of per-image scores (higher = more anomalous).
    labels : np.ndarray
        1-D array of ground-truth integer labels (0-9).
    """
    model.eval()
    scores = []
    labels = []
    for x, y in limit_batches(loader, max_batches):
        x = x.to(device)
        if mode == "true-label":
            # One pass, conditioned on the ground-truth class.
            score = conditional_neg_elbo(model, x, one_hot_labels(y, model.num_classes, device), beta)
        elif mode == "min-over-labels":
            # One pass per trained class; keep the best (lowest) explanation.
            per_label = []
            for candidate in candidate_labels:
                y_c = torch.full((x.size(0),), candidate, dtype=torch.long, device=device)
                per_label.append(conditional_neg_elbo(model, x, one_hot_labels(y_c, model.num_classes, device), beta))
            score = torch.stack(per_label, dim=0).min(dim=0).values
        else:
            raise ValueError(f"Unsupported anomaly score mode: {mode}")
        scores.extend(score.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.asarray(scores), np.asarray(labels)


def run_anomaly_experiment(args, train_dataset, test_dataset, run_dir: Path, device):
    """
    One-class anomaly detection using the CVAE.

    For each digit in --anomaly-digits:
      1. Remove it from training (keep only the other 9 "normal" classes).
      2. Train a CVAE on the normal subset, conditioned on the true labels.
      3. Score all test images with the conditional negative ELBO
         (see --anomaly-score-mode).
      4. Compute AUROC, Average Precision, and recall at the 95th-percentile
         normal threshold (i.e. at a fixed 5 % false-positive rate).
      5. Plot the two score distributions and the decision boundary.
    """
    anomaly_rows = []
    for excluded_digit in args.anomaly_digits:
        # ── Build the normal training set and the full test loader ──
        # The test split deliberately keeps all ten classes: nine normal ones plus
        # the excluded class, which supplies the positive (anomalous) examples.
        normal_train = subset_without_digit(train_dataset, excluded_digit)
        train_loader = make_loader(normal_train, args.batch_size, True,  args.seed, args.num_workers, device)
        test_loader  = make_loader(test_dataset,   args.batch_size, False, args.seed, args.num_workers, device)

        # Labels the model actually sees during training — the only ones that make
        # sense as conditioning candidates at scoring time.
        candidate_labels = [c for c in range(NUM_CLASSES) if c != excluded_digit]

        # ── Train the CVAE on the 9 normal classes ──
        model = ConvCVAE(args.anomaly_latent_dim, num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.anomaly_epochs):
            train_one_epoch(
                model, train_loader, optimizer, device, args.anomaly_loss, args.beta, args.max_train_batches
            )

        # ── Score all test images ──
        scores, labels = anomaly_scores(
            model, test_loader, device, args.beta,
            args.anomaly_score_mode, candidate_labels, args.max_test_batches,
        )
        is_anomaly = (labels == excluded_digit).astype(np.int32)

        # ── Metrics ──
        roc_auc = roc_auc_score(is_anomaly, scores)
        avg_precision = average_precision_score(is_anomaly, scores)
        # Threshold at the 95th percentile of normal scores (operational boundary).
        threshold = np.percentile(scores[is_anomaly == 0], 95)
        predicted = scores >= threshold
        recall_at_95 = (predicted[is_anomaly == 1].mean()).item()

        row = {
            "model": "CVAE",
            "excluded_digit": excluded_digit,
            "train_loss_name": args.anomaly_loss,
            "score_mode": args.anomaly_score_mode,
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
        # density=True normalises both histograms to unit area so the heavily
        # outnumbered anomaly class stays visible next to the normal one.
        plt.figure(figsize=(7, 4))
        plt.hist(scores[is_anomaly == 0], bins=60, alpha=0.7, label="normal", density=True)
        plt.hist(scores[is_anomaly == 1], bins=60, alpha=0.7, label=f"anomaly digit {excluded_digit}", density=True)
        plt.axvline(threshold, color="black", linestyle="--", linewidth=1, label="normal p95")
        plt.xlabel(f"conditional negative ELBO ({args.anomaly_score_mode})")
        plt.ylabel("density")
        plt.title(f"CVAE anomaly detection, excluded digit {excluded_digit}")
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
    Define every command-line flag for the CVAE experiment.

    Relative to ``cvae.py`` the new knobs are ``--samples-per-class`` (width of the
    class-conditional sample grid) and ``--anomaly-score-mode`` (how to pick the
    conditioning label when scoring anomalies).
    """
    parser = argparse.ArgumentParser(description="Conditional VAE experiments for MNIST/Fashion-MNIST.")

    # ── Dataset ──
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST",
                        help="Dataset to train on (default: MNIST).  Both have 10 classes.")
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

    # ── β weighting on the KL term ──
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight on the KL term (default: 1.0).  "
                             "β = 1 → standard CVAE.  β > 1 → stronger regularisation, "
                             "pushing even more class information out of z.  "
                             "β < 1 → more emphasis on reconstruction.")

    # ── CVAE-specific visualisation ──
    parser.add_argument("--samples-per-class", type=int, default=8,
                        help="Columns per class in the class-conditional sample grid (default: 8).  "
                             "Each column shares one latent style across all 10 rows.")

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
                        help="Loss for the anomaly CVAE (default: bce).")
    parser.add_argument("--anomaly-latent-dim", type=int, default=32,
                        help="Bottleneck size for anomaly CVAEs (default: 32).")
    parser.add_argument("--anomaly-epochs", type=int, default=5,
                        help="Training epochs for each anomaly CVAE (default: 5).")
    parser.add_argument("--anomaly-score-mode", choices=["min-over-labels", "true-label"],
                        default="min-over-labels",
                        help="Which label to condition on when scoring test images.  "
                             "'min-over-labels' (default) tries every trained class and keeps the "
                             "best-fitting one, needing no label at test time; 'true-label' uses the "
                             "ground-truth label, which is cheaper but partly measures label novelty.")

    return parser.parse_args()


def main():
    """
    Top-level execution flow:
      1. Parse the CLI, set seeds and pick a device.
      2. Create a run directory that includes beta so different β runs never collide.
      3. Load the data and build loaders (labels now matter — they are model input).
      4. Sweep over (loss, latent_dim); train and log every configuration, saving
         reconstructions, class-conditional samples and label-swap grids.
      5. Save the aggregated CSV and the recon/KL comparison plot.
      6. Optionally run the one-class anomaly-detection experiment.
    """
    # 1. Parse args
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Unique run directory; the "cvae_" prefix keeps it apart from the
    #    unconditional VAE runs produced by cvae.py ("vae_...").
    run_dir = args.out_dir / f"cvae_{args.dataset.lower()}_beta{args.beta}_seed{args.seed}"
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
            _, rows = train_cvae_for_latent_dim(args, latent_dim, loss_name, train_loader, test_loader, run_dir, device)
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
