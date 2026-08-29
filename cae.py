#!/usr/bin/env python3
"""
Convolutional Autoencoder (AE) — Final Experiment Script for MNIST / Fashion-MNIST

This is the consolidated "final" version of the plain (deterministic) autoencoder
pipeline.  It trains a Convolutional Autoencoder across a sweep of configurations
(latent dimensions × reconstruction losses) and evaluates it on:
  1. Reconstruction quality (train / test loss curves, sample grids)
  2. Latent-space structure (2-D scatter plot, latent interpolation strip)
  3. Anomaly detection (one-digit-held-out setup: AUROC, average precision, recall@p95)

Related scripts:
  • cae.py  — the annotated reference AE script (same architecture)
  • cvae.py — the probabilistic counterpart (VAE / β-VAE)

Typical call:
    python Autoencoder_final.py --dataset MNIST --epochs 20 --latent-dims 2 8 32 --losses bce mse
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
import matplotlib.pyplot as plt  # Plotting library (loss curves, scatter plots, histograms)
import numpy as np               # Numerical computing (arrays, percentiles)
import pandas as pd              # Tabular data handling (experiment logs, CSV export)

# PyTorch: deep-learning framework for automatic differentiation & GPU acceleration
import torch
import torch.nn as nn            # Neural-network layers (Conv2d, Linear, BatchNorm2d, ...)
import torch.nn.functional as F  # Stateless functions (activations, loss functions)
from sklearn.metrics import average_precision_score, roc_auc_score   # Anomaly-detection metrics
from torch.utils.data import DataLoader, Subset                      # Batching + index-based subsets
from torchvision import datasets, transforms   # Standard vision datasets / pre-processing pipelines
from torchvision.utils import save_image       # Save a batch of image tensors as a PNG grid
from tqdm import tqdm            # Progress-bar wrapper for loops


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION: Convolutional Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════
# An autoencoder learns to compress (encode) an input image into a low-dimensional
# latent vector z, then decompress (decode) it back into a reconstruction x_hat.
# The encoder is a stack of strided convolutions that shrink the spatial size while
# growing the channel count; the decoder mirrors it with transposed convolutions.
# Training is end-to-end by minimising a pixel-wise reconstruction loss — no labels
# are used anywhere, so this is fully unsupervised representation learning.
# ═══════════════════════════════════════════════════════════════════════════════

class ConvAutoencoder(nn.Module):
    """
    A fully-convolutional autoencoder tailored for 28×28 grayscale images
    (e.g. MNIST, Fashion-MNIST).

    Architecture overview
    ---------------------
      Input  (1, 28, 28)
        → Encoder conv blocks   : 1 → 32 → 64 channels, spatial 28 → 14 → 7
        → Flatten → FC          : 64·7·7 = 3136  →  latent_dim
        → FC → Reshape          : latent_dim  →  64·7·7  →  (64, 7, 7)
        → Decoder transposed-conv blocks : 64 → 32 → 1 channels, spatial 7 → 14 → 28
      Output (1, 28, 28)
    """

    def __init__(self, latent_dim: int):
        """
        Parameters
        ----------
        latent_dim : int
            The size of the bottleneck vector z.  Smaller values force more
            compression (blurrier reconstructions); larger values retain more
            information but risk learning a near-identity mapping.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder (convolutional feature extractor) ─────────────────────────
        # Each Conv2d with stride=2 halves the spatial resolution while expanding
        # the channel count.  BatchNorm normalises activations per channel, which
        # stabilises and speeds up training.  ReLU supplies the non-linearity.
        self.encoder_conv = nn.Sequential(
            # Layer 1:  1 channel → 32 channels,  28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),      # Normalises each of the 32 feature maps independently
            nn.ReLU(inplace=True),   # In-place variant saves a little GPU memory
            # Layer 2:  32 channels → 64 channels,  14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # After the conv stack the tensor shape is (batch, 64, 7, 7);
        # flattened that is 64 * 7 * 7 = 3136 features per image.
        self.encoder_fc = nn.Linear(64 * 7 * 7, latent_dim)

        # ── Decoder (transposed-convolutional upsampler) ──────────────────────
        # Mirror image of the encoder: latent vector → FC → reshape → deconv stack.
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            # Layer 1: 64 channels → 32 channels,  7×7 → 14×14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2: 32 channels → 1 channel,  14×14 → 28×28
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),            # Squeezes every pixel into [0, 1], matching ToTensor()
        )

    # ── Forward helpers (encode / decode / full forward) ──────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Push a batch of images through the encoder to obtain latent vectors.

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape (B, 1, 28, 28), pixel values in [0, 1].

        Returns
        -------
        z : torch.Tensor
            Latent codes of shape (B, latent_dim).
        """
        h = self.encoder_conv(x)     # → (B, 64, 7, 7)
        h = h.flatten(start_dim=1)   # collapse everything after the batch dim: (B, 3136)
        return self.encoder_fc(h)    # → (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a batch of images from their latent codes.

        Parameters
        ----------
        z : torch.Tensor
            Latent codes of shape (B, latent_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed images of shape (B, 1, 28, 28), values in [0, 1].
        """
        h = self.decoder_fc(z)               # → (B, 3136)
        h = h.view(z.size(0), 64, 7, 7)      # reshape back into feature-map form
        return self.decoder_conv(h)          # → (B, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder pass: encode then decode.

        Returns
        -------
        x_hat : reconstructed image tensor, shape (B, 1, 28, 28)
        z     : latent code tensor,         shape (B, latent_dim)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY & HARDWARE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """
    Fix every random-number generator that the pipeline touches so that runs with
    the same seed produce identical weights, losses, and plots.

    This matters for a fair sweep: when only `latent_dim` changes, we want the
    observed difference to come from that change alone — not from a different
    weight initialisation or a different shuffling order.
    """
    random.seed(seed)                     # Python built-in random module
    np.random.seed(seed)                  # NumPy RNG
    torch.manual_seed(seed)               # PyTorch CPU RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU RNG (all devices)
    # CuDNN picks non-deterministic convolution algorithms by default for speed.
    # These two flags force deterministic kernels at a small performance cost.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """
    Return the best available compute device in priority order:
        NVIDIA CUDA GPU  →  Apple Metal (MPS)  →  CPU fallback.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# torchvision.datasets already standardises MNIST / Fashion-MNIST: images are
# 28×28 grayscale PIL images and labels are integers 0-9.  All we add is a
# transform that converts PIL → normalised FloatTensor.
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataset(name: str, root: Path, train: bool):
    """
    Download (if not already cached) and return a torchvision dataset.

    Parameters
    ----------
    name : str
        Must be "MNIST" or "FashionMNIST".
    root : Path
        Local directory where the raw idx files are cached.
    train : bool
        True → the 60 000-image training split.
        False → the 10 000-image test split.

    Returns
    -------
    torchvision.datasets.VisionDataset
    """
    # ToTensor() converts a PIL Image (H×W, uint8 in [0,255]) into a
    # torch.FloatTensor (C×H×W in [0,1]).  For MNIST C = 1 (grayscale).
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
    Wrap a dataset in a PyTorch DataLoader with its own seeded random sampler.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    batch_size : int
        How many samples per SGD step.
    shuffle : bool
        Reshuffle every epoch.  Typically True for training, False for testing
        (so evaluation order — and hence the visualised batch — is deterministic).
    seed : int
        The DataLoader gets its own torch.Generator so that shuffling does not
        consume draws from (and therefore perturb) the global PyTorch RNG.
    num_workers : int
        Number of CPU sub-processes used to load and pre-process batches.
        0 means the main process does everything (slower, but easier to debug).
    device : torch.device
        On CUDA, pin_memory=True keeps page-locked ("pinned") host memory, which
        makes asynchronous CPU→GPU copies substantially faster.

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
    Generator that caps a DataLoader at `max_batches` batches (None = no cap).

    Very useful for fast smoke-tests: pass --max-train-batches 2 to confirm the
    whole pipeline runs end-to-end before launching a full multi-hour sweep.
    """
    for batch_id, batch in enumerate(loader):
        if max_batches is not None and batch_id >= max_batches:
            break
        yield batch


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
# The loss measures how different the reconstruction x_hat is from the original
# image x.  A lower loss means the AE has learned to squeeze the important
# information through the bottleneck and discard the rest.
# ═══════════════════════════════════════════════════════════════════════════════

def ae_loss(x_hat: torch.Tensor, x: torch.Tensor, loss_name: str) -> torch.Tensor:
    """
    Pixel-wise reconstruction loss: summed per image, then averaged over the batch.

    Why `reduction="sum"` then divide by the batch size?
    ---------------------------------------------------
    `reduction="mean"` would also divide by the pixel count (28·28 = 784), making
    the reported number depend on the image resolution.  Summing per image and
    dividing only by the batch size keeps the loss scale resolution-independent,
    so numbers stay comparable across datasets and against the VAE script.

    Parameters
    ----------
    x_hat, x : torch.Tensor
        Tensors of shape (B, 1, 28, 28) with values in [0, 1].
        x_hat is the model output (post-Sigmoid); x is the ground-truth image.
    loss_name : {"bce", "mse"}
        "bce"  — Binary Cross-Entropy: treats each pixel as an independent
                 Bernoulli probability; punishes confident-but-wrong pixels hard.
        "mse"  — Mean-Squared Error: treats reconstruction as regression;
                 punishes large pixel errors quadratically (tends to blur).

    Returns
    -------
    torch.Tensor
        Scalar: average reconstruction loss per image in the batch.
    """
    # Sum over pixels per image, then average over the batch for a stable scale.
    if loss_name == "bce":
        # The decoder ends with Sigmoid, so x_hat already lies in [0, 1] —
        # exactly the range BCE requires and the range ToTensor() produces.
        return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    if loss_name == "mse":
        # MSE would work without the Sigmoid too, but we keep the decoder identical
        # across losses so the only variable in the comparison is the objective.
        return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    # Placeholder for a third objective (L1 / MAE); not wired into --losses yet.
    # if loss_name == "L1":
    #     return F.L1
    raise ValueError(f"Unsupported AE loss: {loss_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION LOOPS
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, loss_name: str, max_batches=None) -> float:
    """
    Run one complete pass (one epoch) over the training data.

    The loop per mini-batch:
      1. Fetch (x, y) from the DataLoader — labels are ignored (unsupervised).
      2. Move x to the compute device.
      3. Clear stale gradients.
      4. Forward pass → reconstruction x_hat.
      5. Compute the reconstruction loss against the original x.
      6. Backward pass (autograd computes ∂loss/∂parameters).
      7. Optimiser step (Adam updates the weights).

    Parameters
    ----------
    model : nn.Module
        The autoencoder.
    loader : DataLoader
        Train-set loader.
    optimizer : torch.optim.Optimizer
        Typically Adam.
    device : torch.device
        Target compute device.
    loss_name : str
        Passed straight through to ae_loss().
    max_batches : int | None
        For debugging: stop the epoch after this many batches.

    Returns
    -------
    float
        Average reconstruction loss per image over the whole epoch.
    """
    model.train()          # Train mode: BatchNorm updates its running statistics
    running = 0.0          # Accumulates (batch loss × batch size) = total loss
    seen = 0               # Total number of images processed this epoch

    # tqdm wraps the iterator with a progress bar and an ETA estimate.
    for x, _ in tqdm(limit_batches(loader, max_batches), desc="train", leave=False):
        x = x.to(device)   # Host → device copy (fast path when pin_memory is on)

        # `set_to_none=True` is faster than writing zeros into the existing
        # gradient buffers — it simply drops the references.
        optimizer.zero_grad(set_to_none=True)
        x_hat, _ = model(x)                       # Forward: image → latent → reconstruction
        loss = ae_loss(x_hat, x, loss_name)       # How bad is the reconstruction?
        loss.backward()                           # Autograd: all parameter gradients
        optimizer.step()                          # Adam updates the weights

        # `loss` is already the *average per image*, so multiplying by the batch
        # size recovers this batch's total.  This keeps the epoch average correct
        # even when the final batch is smaller than batch_size.
        running += loss.item() * x.size(0)
        seen += x.size(0)

    # max(seen, 1) guards against a division by zero on an empty loader.
    return running / max(seen, 1)


@torch.no_grad()          # Decorator disables gradient tracking for the WHOLE function
def evaluate(model, loader, device, loss_name: str, max_batches=None) -> float:
    """
    Run one evaluation pass and return the average reconstruction loss.

    Identical to train_one_epoch except that there is no backward pass and no
    optimiser step.  `@torch.no_grad()` matters here: skipping the autograd graph
    roughly halves memory use and noticeably increases throughput.
    """
    model.eval()           # Eval mode: BatchNorm uses its stored running mean/var
    running = 0.0
    seen = 0
    for x, _ in limit_batches(loader, max_batches):
        x = x.to(device)
        x_hat, _ = model(x)
        loss = ae_loss(x_hat, x, loss_name)
        running += loss.item() * x.size(0)
        seen += x.size(0)
    return running / max(seen, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# These functions write static PNG files so you can inspect what the model
# learned without needing a notebook server.
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_reconstruction_grid(model, loader, device, path: Path, n: int = 16) -> None:
    """
    Save a PNG grid comparing original images against their reconstructions.

    Layout
    ------
    The output file interleaves n sample pairs:
        [orig_0, recon_0, orig_1, recon_1, ...]
    laid out in rows of 8 images.  A sibling file `*_original.png` holds just the
    originals, which is handy for side-by-side figures in a report.

    Parameters
    ----------
    model : ConvAutoencoder
    loader : DataLoader
        Typically the test loader — reconstructions of unseen data show
        generalisation rather than memorisation.
    device : torch.device
    path : Path
        Destination PNG path.  `<stem>_original<suffix>` is auto-created next to it.
    n : int
        Number of images to visualise (a multiple of nrow=8 looks tidiest).
    """
    model.eval()
    x, _ = next(iter(loader))    # Grab the very first batch from the loader
    x = x[:n].to(device)         # Keep only the first n samples

    # Save the reference grid of untouched originals.
    original_path = path.parent / f"{path.stem}_original{path.suffix}"
    save_image(
        x.cpu(),
        original_path,
        nrow=8,
        padding=2
    )

    x_hat, _ = model(x)          # Run the batch through the full autoencoder

    # Interleave originals and reconstructions so each pair sits side-by-side.
    # Resulting tensor shape: (2*n, 1, 28, 28).
    pair_rows = torch.empty((2 * n, 1, 28, 28), device=device)
    pair_rows[0::2] = x          # even indices = originals
    pair_rows[1::2] = x_hat      # odd indices  = reconstructions
    save_image(pair_rows.cpu(), path, nrow=8, padding=2)


@torch.no_grad()
def plot_latent_2d(model, loader, device, path: Path, max_points: int = 5000) -> None:
    """
    Encode test images and scatter-plot their 2-D latent codes, coloured by the
    ground-truth label.  This is only truly meaningful when latent_dim == 2; for
    larger bottlenecks the function would just plot the first two coordinates,
    which carry no special meaning.

    Why visualise the latent space?
    ------------------------------
    A well-trained AE tends to place samples of the same class near each other.
    Heavy overlap between colours means either the bottleneck is too tight or the
    model has not learned discriminative features.  Note the labels are used
    *only* for colouring — training itself never sees them.
    """
    model.eval()
    zs = []      # Collected latent arrays, one per batch
    ys = []      # Collected label arrays, one per batch
    count = 0

    for x, y in loader:
        x = x.to(device)
        z = model.encode(x).cpu().numpy()   # (B, latent_dim)
        zs.append(z)
        ys.append(y.numpy())
        count += x.size(0)
        if count >= max_points:             # Cap the point count to keep plotting fast
            break

    # Concatenate across batches, then trim to exactly max_points.
    z_all = np.concatenate(zs, axis=0)[:max_points]
    y_all = np.concatenate(ys, axis=0)[:max_points]

    # "tab10" is a 10-colour qualitative colormap — one colour per MNIST class.
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, s=7, cmap="tab10", alpha=0.8)
    plt.colorbar(scatter, ticks=list(range(10)))
    plt.title("AE latent space 2D")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def save_interpolation(model, dataset, device, path: Path, steps: int = 11) -> None:
    """
    Linearly interpolate between the latent codes of two images from **different**
    classes and decode every intermediate point.

    Why interpolate?
    ----------------
    A smooth, well-organised latent manifold means that walking a straight line
    between two latent points produces a gradual visual morph.  Abrupt jumps or
    meaningless intermediate frames indicate a fragmented latent space — a known
    weakness of the plain AE relative to a VAE, whose KL term explicitly
    regularises the latent space toward a smooth prior.

    Parameters
    ----------
    model : ConvAutoencoder
    dataset : torch.utils.data.Dataset
        Usually the test dataset, so the endpoints are unseen samples.
    device : torch.device
    path : Path
        PNG save path.  The output is a single row of `steps` images.
    steps : int
        Number of evenly spaced points along the line, endpoints included.
    """
    model.eval()

    # Endpoint A: simply the first sample in the dataset.
    first_x, first_y = dataset[0]

    # Endpoint B: scan forward until we hit a sample from a DIFFERENT class, so the
    # transition is visually interesting rather than two variants of one digit.
    second_x = None
    for x, y in dataset:
        if y != first_y:
            second_x = x
            break
    if second_x is None:
        raise RuntimeError("Could not find two samples with different labels for interpolation.")

    # Add a batch dimension: (1, 1, 28, 28), and move onto the compute device.
    xa = first_x.unsqueeze(0).to(device)
    xb = second_x.unsqueeze(0).to(device)

    za = model.encode(xa)   # (1, latent_dim)
    zb = model.encode(xb)   # (1, latent_dim)

    # `steps` evenly spaced blend factors α ∈ [0, 1], shaped (steps, 1) so they
    # broadcast across the latent dimension:
    #     z(α) = (1 − α)·za + α·zb      (straight-line interpolation)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)   # (steps, 1)
    z = (1 - alphas) * za + alphas * zb                               # (steps, latent_dim)

    decoded = model.decode(z)          # (steps, 1, 28, 28)
    save_image(decoded.cpu(), path, nrow=steps, padding=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER (RECONSTRUCTION SWEEP)
# ═══════════════════════════════════════════════════════════════════════════════

def train_ae_for_latent_dim(args, latent_dim: int, loss_name: str, train_loader, test_loader, run_dir: Path, device):
    """
    Train ONE autoencoder configuration (a specific latent_dim × loss_name pair).

    `main()` calls this in a nested loop so every combination is trained,
    evaluated, and has its artefacts saved independently of the others.

    Returns
    -------
    model : ConvAutoencoder
        The trained model (returned in case a caller wants further analysis).
    rows : list[dict]
        One dictionary per epoch, ready to be turned into a DataFrame.
    """
    # A fresh model per configuration — never reuse the previous run's weights.
    model = ConvAutoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_name, args.max_train_batches)
        test_loss = evaluate(model, test_loader, device, loss_name, args.max_test_batches)

        # Log epoch-level metrics.  Carrying "model" and "dataset" in every row
        # makes the CSV self-describing when logs from several runs are merged.
        row = {
            "model": "AE",
            "dataset": args.dataset,
            "loss_name": loss_name,
            "latent_dim": latent_dim,
            "epoch": epoch,
            "train_recon_loss": train_loss,   # average loss on the training split
            "test_recon_loss": test_loss,     # average loss on the held-out test split
            "seed": args.seed,
        }
        rows.append(row)
        print(row)                            # Live progress on stdout

    # ── Save this configuration's artefacts ──────────────────────────────────

    # 1. Model checkpoint.  We store only the state_dict (weights + buffers)
    #    rather than the pickled model object: it is smaller and does not couple
    #    the file to the exact Python / PyTorch version used here.
    torch.save(model.state_dict(), run_dir / f"ae_{loss_name}_latent_{latent_dim}.pt")

    # 2. Visual grid of reconstructions vs. originals.
    save_reconstruction_grid(model, test_loader, device, run_dir / f"recon_{loss_name}_latent_{latent_dim}.png")

    # 3. The 2-D scatter and the interpolation strip are only meaningful when the
    #    bottleneck is exactly 2-D; for higher dims a scatter of the first two
    #    arbitrary coordinates would be misleading.
    if latent_dim == 2:
        plot_latent_2d(model, test_loader, device, run_dir / f"latent_map_2d_{loss_name}.png")
        save_interpolation(model, test_loader.dataset, device, run_dir / f"interpolation_ae_{loss_name}.png")
    return model, rows


def save_loss_comparison(all_rows: list[dict], run_dir: Path) -> None:
    """
    After every configuration has been trained, build:
      1. A summary CSV holding the *final-epoch* test loss of each
         (loss_name, latent_dim) pair.
      2. A figure with one subplot per loss function, showing how the test
         reconstruction error changes as the bottleneck grows.

    The x-axis uses a log₂ scale because the default latent_dims are powers of
    two (2, 8, 32, 128), which then space out evenly.
    """
    df = pd.DataFrame(all_rows)
    # Group by unique (loss_name, latent_dim) and keep only the LAST epoch of each
    # group — i.e. the fully-trained model rather than an intermediate checkpoint.
    final_df = df.sort_values("epoch").groupby(["loss_name", "latent_dim"], as_index=False).tail(1)
    final_df = final_df.sort_values(["loss_name", "latent_dim"])
    final_df.to_csv(run_dir / "loss_comparison_summary.csv", index=False)

    loss_names = final_df["loss_name"].unique().tolist()
    # One subplot per loss function, arranged horizontally.
    # squeeze=False guarantees `axes` stays 2-D even for a single loss function.
    fig, axes = plt.subplots(1, len(loss_names), figsize=(6 * len(loss_names), 4), squeeze=False)
    for ax, loss_name in zip(axes[0], loss_names):
        loss_df = final_df[final_df["loss_name"] == loss_name]
        ax.plot(loss_df["latent_dim"], loss_df["test_recon_loss"], marker="o")
        ax.set_xscale("log", base=2)                     # Logarithmic x-axis, base 2
        # Force ticks at exactly the sampled latent dims (not log-scale defaults).
        ax.set_xticks(loss_df["latent_dim"].tolist())
        ax.set_xticklabels([str(v) for v in loss_df["latent_dim"].tolist()])
        ax.set_xlabel("latent dim")
        ax.set_ylabel(f"test reconstruction {loss_name.upper()}")
        ax.set_title(f"AE trained with {loss_name.upper()}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_comparison_by_latent_dim.png", dpi=160)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY-DETECTION EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════
# Core idea: an AE only learns to faithfully reconstruct the distribution it was
# trained on.  If we withhold one digit class during training, that class becomes
# an "anomaly" at test time — the model has never seen it, so its reconstruction
# will be poor and its reconstruction error high.
#
# Reconstruction error therefore acts as an anomaly SCORE (higher = more anomalous).
# We rank test samples by that score and compute standard detection metrics
# (AUROC, average precision) plus one operational metric at a fixed threshold.
# ═══════════════════════════════════════════════════════════════════════════════

def subset_without_digit(dataset, excluded_digit: int) -> Subset:
    """
    Return a Subset containing every sample whose label is NOT `excluded_digit`.

    This is the "normal" training data: the AE learns the distribution of the
    remaining nine digits and will struggle to reconstruct the excluded one.
    """
    indices = []
    for idx, (_, y) in enumerate(dataset):
        if int(y) != excluded_digit:
            indices.append(idx)
    return Subset(dataset, indices)


@torch.no_grad()
def reconstruction_errors(model, loader, device, max_batches=None):
    """
    Compute the reconstruction BCE for every image in the loader.

    We take the per-image SUM of the pixel-wise BCE (not the mean); that sum is
    the anomaly score, with larger values meaning worse reconstruction.  BCE is
    used here regardless of the training loss so that scores stay comparable
    across runs.

    Returns
    -------
    errors : np.ndarray
        1-D array of per-image anomaly scores.
    labels : np.ndarray
        1-D array of ground-truth integer labels (0-9).
    """
    model.eval()
    errors = []
    labels = []
    for x, y in limit_batches(loader, max_batches):
        x = x.to(device)
        x_hat, _ = model(x)
        # reduction="none" keeps the per-pixel losses; flatten(1) merges the
        # channel/height/width dims so sum(dim=1) totals each image separately.
        err = F.binary_cross_entropy(x_hat, x, reduction="none").flatten(1).sum(dim=1)
        errors.extend(err.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.asarray(errors), np.asarray(labels)


def run_anomaly_experiment(args, train_dataset, test_dataset, run_dir: Path, device):
    """
    Run the one-class anomaly-detection experiment, looping over
    `args.anomaly_digits`.  For each digit:

      1. Remove that digit from the training set.
      2. Train a fresh AE on the remaining nine digits.
      3. Score every test image by its reconstruction error under that AE.
      4. Compute:
            AUROC             – area under the ROC curve (threshold-agnostic ranking quality)
            Average Precision – area under the precision–recall curve, the more
                                informative metric here because the classes are
                                imbalanced (≈9 normal digits vs. 1 anomalous)
            recall_at_95      – what fraction of true anomalies is caught when the
                                threshold sits at the 95th percentile of normal
                                scores (i.e. at a fixed 5 % false-positive rate)
      5. Plot and save the two score distributions with the decision threshold.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI flags (contains the anomaly hyper-parameters).
    train_dataset, test_dataset : torchvision.datasets.*
    run_dir : Path
        Output directory for the PNG plots and the CSV log.
    device : torch.device

    Returns
    -------
    list[dict]
        Summary rows, one per excluded digit.
    """
    anomaly_rows = []
    for excluded_digit in args.anomaly_digits:
        # ── Step 1: build the "normal" training set (and the full test loader) ──
        # The test set intentionally keeps ALL ten digits: nine normal classes plus
        # the excluded one, which supplies the positive (anomalous) examples.
        normal_train = subset_without_digit(train_dataset, excluded_digit)
        train_loader = make_loader(normal_train, args.batch_size, True, args.seed, args.num_workers, device)
        test_loader = make_loader(test_dataset, args.batch_size, False, args.seed, args.num_workers, device)

        # ── Step 2: train a fresh AE on the nine normal digits only ──
        model = ConvAutoencoder(args.anomaly_latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.anomaly_epochs):
            train_one_epoch(model, train_loader, optimizer, device, args.anomaly_loss, args.max_train_batches)

        # ── Step 3: score every test image ──
        errors, labels = reconstruction_errors(model, test_loader, device, args.max_test_batches)
        # Ground-truth anomaly flag: 1 for the excluded digit, 0 for the nine normal ones.
        is_anomaly = (labels == excluded_digit).astype(np.int32)

        # ── Step 4: metrics ──
        roc_auc = roc_auc_score(is_anomaly, errors)
        avg_precision = average_precision_score(is_anomaly, errors)
        # Operational threshold: the 95th percentile of the NORMAL scores, i.e. the
        # point where we accept a 5 % false-positive rate on known-good data.
        threshold = np.percentile(errors[is_anomaly == 0], 95)
        predicted = errors >= threshold
        # Fraction of true anomalies whose score exceeds that threshold (= recall).
        recall_at_95 = (predicted[is_anomaly == 1].mean()).item()

        row = {
            "excluded_digit": excluded_digit,
            "train_loss_name": args.anomaly_loss,
            "latent_dim": args.anomaly_latent_dim,
            "epochs": args.anomaly_epochs,
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "normal_p95_threshold": threshold,
            "anomaly_recall_at_normal_p95": recall_at_95,
            "seed": args.seed,
        }
        anomaly_rows.append(row)
        print(row)

        # ── Step 5: plot the score distributions ──
        # density=True normalises both histograms to unit area, so the heavily
        # outnumbered anomaly class stays visible next to the normal one.
        plt.figure(figsize=(7, 4))
        plt.hist(errors[is_anomaly == 0], bins=60, alpha=0.7, label="normal", density=True)
        plt.hist(errors[is_anomaly == 1], bins=60, alpha=0.7, label=f"anomaly digit {excluded_digit}", density=True)
        # Dashed vertical line marking the 95th percentile of the normal scores.
        plt.axvline(threshold, color="black", linestyle="--", linewidth=1, label="normal p95")
        plt.xlabel("reconstruction BCE")
        plt.ylabel("density")
        plt.title(f"AE anomaly detection, excluded digit {excluded_digit}")
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
    Build the ArgumentParser and return the parsed CLI flags.
    Run ``python Autoencoder_final.py --help`` for the generated help text.
    """
    parser = argparse.ArgumentParser(description="Autoencoder experiments for MNIST/Fashion-MNIST.")

    # ── Dataset selection & I/O paths ──
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST")   # Dataset to train on
    parser.add_argument("--data-dir", type=Path, default=Path("data"))                     # Cache dir for raw downloads
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))                      # Root dir for run artefacts

    # ── Reconstruction-sweep training hyper-parameters ──
    parser.add_argument("--epochs", type=int, default=10)                                  # Epochs per configuration
    parser.add_argument("--batch-size", type=int, default=128)                             # SGD mini-batch size
    parser.add_argument("--lr", type=float, default=1e-3)                                  # Adam learning rate
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[2, 8, 32, 128])     # Bottleneck sizes to sweep
    parser.add_argument("--losses", choices=["bce", "mse"], nargs="+", default=["bce", "mse"])  # Losses to compare

    # ── Infrastructure / reproducibility ──
    parser.add_argument("--seed", type=int, default=42)                                    # Global RNG seed
    parser.add_argument("--num-workers", type=int, default=2)                              # DataLoader sub-processes (0 to debug)
    parser.add_argument("--max-train-batches", type=int, default=None)                     # Cap train batches (None = all)
    parser.add_argument("--max-test-batches", type=int, default=None)                      # Cap test batches (None = all)

    # ── Anomaly-detection sub-experiment ──
    parser.add_argument("--skip-anomaly", action="store_true")                             # Skip the anomaly stage entirely
    parser.add_argument("--anomaly-digits", type=int, nargs="+", default=list(range(10)))  # Digits held out, one at a time
    parser.add_argument("--anomaly-loss", choices=["bce", "mse"], default="bce")           # Loss for the anomaly AEs
    parser.add_argument("--anomaly-latent-dim", type=int, default=32)                      # Bottleneck for the anomaly AEs
    parser.add_argument("--anomaly-epochs", type=int, default=5)                           # Epochs per anomaly AE
    return parser.parse_args()


def main():
    """
    Main execution pipeline:

      1. Parse the CLI flags.
      2. Fix the RNG seeds and pick the best available device.
      3. Create a unique run directory inside ``--out-dir``.
      4. Load the train / test datasets and build their DataLoaders.
      5. For every (loss, latent_dim) combination:
            - train the AE for N epochs,
            - log the per-epoch losses,
            - save the checkpoint, the reconstruction grid and — when
              latent_dim == 2 — the 2-D latent scatter plus interpolation strip.
      6. Aggregate all logs into a CSV and a comparison plot.
      7. Unless --skip-anomaly is set, run the one-class anomaly experiment for
         every digit listed in --anomaly-digits.
    """
    # ── 1-2. Args, reproducibility, device ──
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ── 3. Run directory, e.g. ``runs/ae_mnist_seed42/`` ──
    # The dataset and seed are baked into the name so parallel runs never collide.
    run_dir = args.out_dir / f"ae_{args.dataset.lower()}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Data ──
    train_dataset = get_dataset(args.dataset, args.data_dir, train=True)
    test_dataset = get_dataset(args.dataset, args.data_dir, train=False)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed, args.num_workers, device)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.seed, args.num_workers, device)

    # ── 5. Reconstruction sweep over every (loss, latent_dim) pair ──
    all_rows = []
    for loss_name in args.losses:
        for latent_dim in args.latent_dims:
            _, rows = train_ae_for_latent_dim(args, latent_dim, loss_name, train_loader, test_loader, run_dir, device)
            all_rows.extend(rows)

    # ── 6. Save the aggregated epoch log and the comparison plot ──
    pd.DataFrame(all_rows).to_csv(run_dir / "experiment_log.csv", index=False)
    save_loss_comparison(all_rows, run_dir)

    # ── 7. Optional anomaly-detection experiment ──
    if not args.skip_anomaly:
        anomaly_rows = run_anomaly_experiment(args, train_dataset, test_dataset, run_dir, device)
        pd.DataFrame(anomaly_rows).to_csv(run_dir / "anomaly_log.csv", index=False)

    print(f"Done. Results saved to: {run_dir.resolve()}")


# Python entry-point guard: run main() only when this file is executed directly,
# not when it is imported.  That lets another script ``import Autoencoder_final``
# and reuse ConvAutoencoder or the helper functions without triggering training.
if __name__ == "__main__":
    main()
