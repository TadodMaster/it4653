#!/usr/bin/env python3
"""
Convolutional Autoencoder (AE) Experiments on MNIST / Fashion-MNIST

This script trains a Convolutional Autoencoder with various configurations
(latent dimensions, loss functions) and evaluates it on:
  1. Reconstruction quality (train / test loss curves, sample grids)
  2. Latent-space visualisation (2-D scatter, interpolation)
  3. Anomaly detection (one-digit-removed setup, AUROC, precision–recall)

Typical call:
    python cae.py --dataset MNIST --epochs 20 --latent-dims 2 8 32 --losses bce mse
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

# PyTorch: deep-learning framework for automatic differentiation & GPU acceleration
import torch
import torch.nn as nn            # Neural-network layers (Linear, Conv2d, BatchNorm2d, etc.)
import torch.nn.functional as F  # Stateless functions (activations, loss functions, interpolation)
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
torchvision = None               # placeholder (filled below)
from torchvision import datasets, transforms   # Standard vision datasets / image pre-processing pipelines
from torchvision.utils import save_image       # Save a grid of images in tensor form to a PNG file
from tqdm import tqdm            # Progress-bar wrapper for loops


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION: Convolutional Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════
# An autoencoder learns to compress (encode) input data into a low-dimensional
# latent representation z, then decompress (decode) it back to reconstruct the
# original input.  The encoder is a stack of convolutional layers that reduce
# spatial size while increasing channels; the decoder reverses this process using
# transposed convolutions.  The model is trained end-to-end by minimising a
# reconstruction loss.
# ═══════════════════════════════════════════════════════════════════════════════

class ConvAutoencoder(nn.Module):
    """
    A fully-convolutional autoencoder tailored for 28×28 grayscale images
    (e.g. MNIST, Fashion-MNIST).

    Architecture overview
    -----------------------
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
            compression; larger values retain more information.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder (convolutional feature extractor) ─────────────────────
        # Each Conv2d halves spatial dimensions (stride=2) while expanding channels.
        # BatchNorm normalises activations per channel to speed up training.
        # ReLU introduces non-linearity needed to learn complex mappings.
        self.encoder_conv = nn.Sequential(
            # Layer 1:  1  channel → 32 channels,  28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),       # Normalises each of the 32 feature maps independently
            nn.ReLU(inplace=True),  # In-place variant saves a small amount of GPU memory
            # Layer 2:  32 channels → 64 channels,  14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # After conv layers the tensor shape is (batch, 64, 7, 7).
        # Flattened that is 64 * 7 * 7 = 3136 elements per image.
        self.encoder_fc = nn.Linear(64 * 7 * 7, latent_dim)

        # ── Decoder (transposed-convolutional upsampler) ──────────────────
        # mirror image of the encoder: latent vector → FC → reshape → deconv
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            # Layer 1: 64 channels → 32 channels,  7×7 → 14×14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2: 32 channels → 1 channel,  14×14 → 28×28
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),            # Squeezes every pixel to the range [0, 1]
        )

    # ── Forward helpers (encode / decode / full forward) ────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Push a batch of images through the encoder to obtain latent vectors.

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape (batch_size, 1, 28, 28), pixel values in [0, 1].

        Returns
        -------
        z : torch.Tensor
            Latent codes of shape (batch_size, latent_dim).
        """
        h = self.encoder_conv(x)     # → (B, 64, 7, 7)
        h = h.flatten(start_dim=1)   # collapse all dims after batch: (B, 64*7*7)
        return self.encoder_fc(h)    # → (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a batch of images from their latent codes.

        Parameters
        ----------
        z : torch.Tensor
            Latent codes of shape (batch_size, latent_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed images of shape (batch_size, 1, 28, 28).
        """
        h = self.decoder_fc(z)                   # → (B, 64*7*7)
        h = h.view(z.size(0), 64, 7, 7)         # reshape back to feature-map form
        return self.decoder_conv(h)              # → (B, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder pass: encode then decode.

        Returns
        -------
        x_hat : reconstructed image tensor, shape (B, 1, 28, 28)
        z     : latent code tensor,          shape (B, latent_dim)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY & HARDWARE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """
    Fix every random-number generator that the pipeline touches so that
    runs with the same seed produce identical weights, losses, and plots.

    This is crucial for fair comparison: if you change one hyper-parameter
    (e.g. latent_dim) you want the only difference in the result to come from
    that change, not from different random initialisations or data shuffling.
    """
    random.seed(seed)                # Python built-in random module
    np.random.seed(seed)             # NumPy random module
    torch.manual_seed(seed)          # PyTorch CPU RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU RNG (all devices)
    # CuDNN is a highly-optimised CUDA backend for convolutions.  It uses
    # non-deterministic algorithms by default for speed.  These flags force
    # deterministic algorithms at the cost of a small performance penalty.
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
# The torchvision.datasets module already standardises MNIST / Fashion-MNIST:
# images are 28×28 grayscale PIL images and labels are integers 0-9.
# We only need to add a transform that converts PIL → normalised FloatTensor.
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataset(name: str, root: Path, train: bool):
    """
    Download (if not already cached) and return a torchvision dataset.

    Parameters
    ----------
    name : str
        Must be "MNIST" or "FashionMNIST".
    root : Path
        Local directory where raw data (binary idx files) will be cached.
    train : bool
        True → the 60 000-image training split.
        False → the 10 000-image test split.

    Returns
    -------
    torchvision.datasets.VisionDataset
    """
    # transforms.ToTensor() converts a PIL Image (H×W in [0,255]) into a
    # torch.FloatTensor (C×H×W in [0,1]).  For MNIST C=1 (grayscale).
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
    Wrap a dataset in a PyTorch DataLoader with a seeded random sampler.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    batch_size : int
        How many samples per SGD step.
    shuffle : bool
        Whether to reshuffle the data at every epoch.  Typically True for
        training, False for testing (to keep evaluation deterministic).
    seed : int
        The DataLoader needs its own torch.Generator so shuffling does not
        interfere with the global PyTorch RNG.
    num_workers : int
        Number of CPU sub-processes used to load and pre-process batches.
        0 means the main process does everything (slower but easier to debug).
    device : torch.device
        When on CUDA, pin_memory=True causes the DataLoader to keep page-locked
        ("pinned") CPU memory, which accelerates asynchronous CPU→GPU copies.

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
    Wrapper that caps a DataLoader at `max_batches` batches.
    Very useful for fast smoke-tests: set --max-train-batches 2 to verify the
    pipeline does not crash before launching a full multi-hour run.
    """
    for batch_id, batch in enumerate(loader):
        if max_batches is not None and batch_id >= max_batches:
            break
        yield batch


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
# The job of the loss is to measure how different the reconstruction x_hat is
# from the original image x.  Lower loss ⇒ the AE has learned to compress the
# important information and discard noise.
# ═══════════════════════════════════════════════════════════════════════════════

def ae_loss(x_hat: torch.Tensor, x: torch.Tensor, loss_name: str) -> torch.Tensor:
    """
    Pixel-wise reconstruction loss, averaged per image then averaged over the batch.

    Why `reduction="sum"` per image?
    -----------------------------
    Using `mean` would divide by the total number of pixels (28·28 = 784).
    If we later compare different image sizes, losses would not be comparable.
    Summing per image and dividing by the batch size keeps the loss scale
    independent of the image resolution.

    Parameters
    ----------
    x_hat, x : torch.Tensor
        Tensors of shape (B, 1, 28, 28) in the range [0, 1].
        x_hat is the model output (after Sigmoid), x is the ground-truth image.
    loss_name : {"bce", "mse"}
        "bce"  — Binary Cross-Entropy: good when pixels are treated as independent
                 Bernoulli probabilities.  Penalises confident but wrong pixels heavily.
        "mse"  — Mean-Squared Error:   treats reconstruction as regression.
                 Penalises large pixel errors quadratically.

    Returns
    -------
    torch.Tensor
        Scalar tensor holding the average reconstruction loss per image in the batch.
    """
    if loss_name == "bce":
        # Since the decoder ends with Sigmoid, x_hat is already in [0, 1],
        # matching the ground-truth range of ToTensor().
        return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    if loss_name == "mse":
        # MSE works well even without a Sigmoid, but here we keep it consistent.
        return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    raise ValueError(f"Unsupported AE loss: {loss_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION LOOPS
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, loss_name: str, max_batches=None) -> float:
    """
    Run one complete pass (one epoch) over the training data.

    The loop:
      1. Fetch a mini-batch (x, y) from the DataLoader.
      2. Move x to GPU.
      3. Zero existing gradients.
      4. Forward pass → get reconstruction x_hat.
      5. Compute loss between x_hat and original x.
      6. Backward pass ( computes ∂loss/∂parameters ).
      7. Optimiser step (Adam updates weights).

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
        Passed to ae_loss().
    max_batches : int | None
        For debugging: stop after this many batches.

    Returns
    -------
    float
        Average reconstruction loss per image over the epoch.
    """
    model.train()                    # Training mode: BatchNorm uses running stats, enables dropout
    running = 0.0                    # Accumulate total loss × samples (numerically more stable than averaging iteratively)
    seen = 0                         # Total number of images processed this epoch

    # tqdm adds a nice progress bar with estimated time remaining.
    for x, _ in tqdm(limit_batches(loader, max_batches), desc="train", leave=False):
        x = x.to(device)             # Non-blocking copy when pin_memory=True and CUDA is used.

        # `set_to_none=True` is faster than zero_grad() because it does not write
        # zeros into existing gradient buffers; it replaces them with None.
        optimizer.zero_grad(set_to_none=True)
        x_hat, _ = model(x)                       # Forward: image → latent → reconstructed image
        loss = ae_loss(x_hat, x, loss_name)       # How bad is the reconstruction?
        loss.backward()                           # Autograd: compute all parameter gradients
        optimizer.step()                            # Update weights via Adam

        # We multiply by batch size because `reduction="sum"` already made `loss`
        # the *average* loss.  Multiplying brings it back to the total loss of this
        # batch, so we can average correctly across variable-sized last batches.
        running += loss.item() * x.size(0)
        seen += x.size(0)

    # Guard against division by zero on an empty loader.
    return running / max(seen, 1)


@torch.no_grad()                     # Decorator disables gradient tracking for the ENTIRE function.
def evaluate(model, loader, device, loss_name: str, max_batches=None) -> float:
    """
    Run one evaluation pass and return the average reconstruction loss.

    Identical to train_one_epoch except there is no back-propagation or optimiser
    step.  `@torch.no_grad()` is critical here because it halves memory usage
    and doubles throughput by skipping the autograd bookkeeping entirely.
    """
    model.eval()                     # Evaluation mode: BatchNorm uses running means/vars (no update)
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
# These functions save static PNG images that let you inspect what the model
# has learned without running a notebook server.
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_reconstruction_grid(model, loader, device, path: Path, n: int = 16) -> None:
    """
    Save a PNG grid comparing original images to their reconstructions.

    Layout
    ------
    The output file shows n sample pairs interleaved:
        [orig_0, recon_0, orig_1, recon_1, ...]
    arranged in rows of 8 images.
    A separate file `*_original.png` contains just the originals.

    Parameters
    ----------
    model : ConvAutoencoder
    loader : DataLoader
        Typically the test loader (unseen data shows generalisation).
    device : torch.device
    path : Path
        Destination PNG path.  A sibling file `_original.png` is auto-created.
    n : int
        Number of images to visualise (must be divisible by `nrow` for aesthetics).
    """
    model.eval()
    x, _ = next(iter(loader))        # Grab the very first batch from the loader
    x = x[:n].to(device)             # Keep only the first n samples

    # Save a reference grid of the original images
    original_path = path.parent / f"{path.stem}_original{path.suffix}"
    save_image(x.cpu(), original_path, nrow=8, padding=2, normalize=False)

    x_hat, _ = model(x)              # Run through the full AE

    # Interleave original and reconstruction so they appear side-by-side in pairs
    # Result tensor shape: (2*n, 1, 28, 28)
    pair_rows = torch.empty((2 * n, 1, 28, 28), device=device)
    pair_rows[0::2] = x             # even indices = originals
    pair_rows[1::2] = x_hat         # odd indices  = reconstructions
    save_image(pair_rows.cpu(), path, nrow=8, padding=2)


@torch.no_grad()
def plot_latent_2d(model, loader, device, path: Path, max_points: int = 5000) -> None:
    """
    Encode test images and scatter-plot their 2-D latent codes, colour-coded by
    ground-truth label.  Only makes real sense when latent_dim == 2, but the
    function blindly plots the first two dimensions otherwise.

    Why visualise the latent space?
    ---------------------------------------------------------------
    A well-trained AE should cluster the latent codes of the same class together.
    If classes overlap heavily, the bottleneck is either too small or the AE
    has not learned discriminative features.
    """
    model.eval()
    zs = []      # Collector for latent numpy arrays
    ys = []      # Collector for label arrays
    count = 0

    for x, y in loader:
        x = x.to(device)
        z = model.encode(x).cpu().numpy()   # (B, latent_dim)
        zs.append(z)
        ys.append(y.numpy())
        count += x.size(0)
        if count >= max_points:            # Cap to keep plotting fast
            break

    # Concatenate across batches and slice to the cap
    z_all = np.concatenate(zs, axis=0)[:max_points]
    y_all = np.concatenate(ys, axis=0)[:max_points]

    # tab10 is a 10-colour map: perfect for the 10 MNIST classes
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
    Linearly interpolate between the latent codes of two images from
    **different** classes and decode each intermediate point.

    Why interpolate?
    ----------------
    A smooth, meaningful latent manifold implies that walking a straight line
    between two latent points produces a gradual visual morph.  Sudden jumps
    or meaningless intermediate images indicate a fragmented latent space.

    Parameters
    ----------
    model : ConvAutoencoder
    dataset : torch.utils.data.Dataset
        Usually the test dataset for generalisation.
    device : torch.device
    path : Path
        PNG save path.  The output is a single row of `steps` images.
    steps : int
        Number of evenly-spaced points along the interpolation line, including
        both endpoints.
    """
    model.eval()

    # Pick the very first sample
    first_x, first_y = dataset[0]

    # Scan forward until we find a sample from a DIFFERENT class.
    # This guarantees a visually interesting transition, not an interpolation
    # between two variants of the same digit.
    second_x = None
    for x, y in dataset:
        if y != first_y:
            second_x = x
            break
    if second_x is None:
        raise RuntimeError("Could not find two samples with different labels for interpolation.")

    # Add batch dimension: (1, 1, 28, 28) and move to compute device
    xa = first_x.unsqueeze(0).to(device)
    xb = second_x.unsqueeze(0).to(device)

    za = model.encode(xa)   # (1, latent_dim)
    zb = model.encode(xb)

    # Create `steps` evenly-spaced scalars α ∈ [0, 1]
    # z_interp(α) = (1-α)·za + α·zb   (linear interpolation in Euclidean space)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)   # (steps, 1)
    z = (1 - alphas) * za + alphas * zb                                # (steps, latent_dim)

    decoded = model.decode(z)          # (steps, 1, 28, 28)
    save_image(decoded.cpu(), path, nrow=steps, padding=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER (RECONSTRUCTION SWEEP)
# ═══════════════════════════════════════════════════════════════════════════════

def train_ae_for_latent_dim(args, latent_dim: int, loss_name: str, train_loader, test_loader, run_dir: Path, device):
    """
    Train ONE autoencoder configuration (a specific latent_dim and loss_name).

    The function is called in a nested loop in `main()` so that every
    (loss_name × latent_dim) combination is trained, evaluated, and its
    artefacts saved independently.

    Returns
    -------
    model : ConvAutoencoder
        The trained model (useful for further downstream tasks).
    rows : list[dict]
        One dictionary per epoch, ready to be concatenated into a DataFrame.
    """
    # Fresh model for every configuration (do not reuse previous weights)
    model = ConvAutoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_name, args.max_train_batches)
        test_loss  = evaluate(model, test_loader, device, loss_name, args.max_test_batches)

        # Log epoch-level metrics.  The "model" and "dataset" columns make this
        # CSV self-describing if you later concatenate logs from many experiments.
        row = {
            "model": "AE",
            "dataset": args.dataset,
            "loss_name": loss_name,
            "latent_dim": latent_dim,
            "epoch": epoch,
            "train_recon_loss": train_loss,   # average loss on the training set
            "test_recon_loss": test_loss,    # average loss on the held-out test set
            "seed": args.seed,
        }
        rows.append(row)
        print(row)                         # Print to stdout so the user sees live progress

    # ── Save configuration-specific artefacts ──

    # 1. Model checkpoint.  We save only the state_dict (weights + buffers),
    # not the full model object, because state_dicts are smaller and do not
    # couple to the exact Python / PyTorch versions.
    torch.save(model.state_dict(), run_dir / f"ae_{loss_name}_latent_{latent_dim}.pt")

    # 2. Visual grid of reconstructions vs. originals.
    save_reconstruction_grid(model, test_loader, device, run_dir / f"recon_{loss_name}_latent_{latent_dim}.png")

    # 3. 2-D latent visualisation and interpolation are meaningful ONLY when
    #    the bottleneck is exactly 2-D.  For higher dimensions the scatter plot
    #    would just show the first two arbitrary coordinates, which is misleading.
    if latent_dim == 2:
        plot_latent_2d(model, test_loader, device, run_dir / f"latent_map_2d_{loss_name}.png")
        save_interpolation(model, test_loader.dataset, device, run_dir / f"interpolation_ae_{loss_name}.png")

    return model, rows


def save_loss_comparison(all_rows: list[dict], run_dir: Path) -> None:
    """
    After every configuration has been trained, build:
      1. A summary CSV with the *final-epoch* test loss for each (loss, latent_dim) pair.
      2. A plot with one subplot per loss function, showing how test reconstruction
         error changes as the bottleneck size grows.

    The plot uses a log₂ x-axis because the default latent_dims are powers of two
    (2, 8, 32, 128), making the spacing linear in a log scale.
    """
    df = pd.DataFrame(all_rows)
    # Group by unique (loss_name, latent_dim) configurations and keep only the LAST
    # epoch of each group (the fully converged model).
    final_df = df.sort_values("epoch").groupby(["loss_name", "latent_dim"], as_index=False).tail(1)
    final_df = final_df.sort_values(["loss_name", "latent_dim"])
    final_df.to_csv(run_dir / "loss_comparison_summary.csv", index=False)

    loss_names = final_df["loss_name"].unique().tolist()
    # One subplot per loss function, arranged horizontally.
    fig, axes = plt.subplots(1, len(loss_names), figsize=(6 * len(loss_names), 4), squeeze=False)

    for ax, loss_name in zip(axes[0], loss_names):
        loss_df = final_df[final_df["loss_name"] == loss_name]
        ax.plot(loss_df["latent_dim"], loss_df["test_recon_loss"], marker="o")
        ax.set_xscale("log", base=2)                     # Logarithmic scale in base 2
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
# Core idea: an AE learns to faithfully reconstruct the data distribution it was
# trained on.  If we withhold one digit class during training, that class
# becomes an "anomaly" at test time.  Because the AE has never seen that class,
# its reconstructions will be poor → high reconstruction error.
#
# Thus reconstruction error acts as an anomaly SCORE: high score = likely anomaly.
# We threshold this score and compute standard classification metrics (AUROC, AP).
# ═══════════════════════════════════════════════════════════════════════════════

def subset_without_digit(dataset, excluded_digit: int) -> Subset:
    """
    Return a Subset containing every sample whose label is NOT `excluded_digit`.

    This becomes the "normal" training data: the AE learns the distribution of
    the other 9 digits and will struggle to reconstruct the excluded one.
    """
    indices = []
    for idx, (_, y) in enumerate(dataset):
        if int(y) != excluded_digit:
            indices.append(idx)
    return Subset(dataset, indices)


@torch.no_grad()
def reconstruction_errors(model, loader, device, max_batches=None):
    """
    Compute reconstruction BCE for every image in the loader.

    We compute the per-image SUM of pixel-wise BCE (not mean).  This sum is the
    anomaly score: larger values indicate worse reconstruction quality.

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
        # reduction="none" gives per-pixel losses; flatten then sum over all pixels
        err = F.binary_cross_entropy(x_hat, x, reduction="none").flatten(1).sum(dim=1)
        errors.extend(err.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.asarray(errors), np.asarray(labels)


def run_anomaly_experiment(args, train_dataset, test_dataset, run_dir: Path, device):
    """
    Run the one-class anomaly-detection experiment in a loop over
    `args.anomaly_digits`.  For each digit:

      1. Remove it from the training set.
      2. Train a fresh AE on the remaining 9 digits.
      3. Score every test image by its reconstruction error on this AE.
      4. Compute:
            AUROC              – area under the ROC curve (threshold-agnostic)
            Average Precision  – area under the precision-recall curve (good for
                                 imbalanced classes since we have 9 normal vs. 1 anomaly)
            recall_at_95      – what fraction of true anomalies are caught when the
                                 threshold is set at the 95th percentile of normal scores?
      5. Plot and save the score distributions with the decision threshold.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI flags (contains anomaly hyper-parameters).
    train_dataset, test_dataset : torchvision.datasets.*
    run_dir : Path
        Output directory for PNG plots and CSV log.
    device : torch.device

    Returns
    -------
    list[dict]
        Summary rows, one per excluded digit.
    """
    anomaly_rows = []

    for excluded_digit in args.anomaly_digits:
        # ── Step 1: Build the "normal" training set ──
        normal_train = subset_without_digit(train_dataset, excluded_digit)
        train_loader = make_loader(normal_train, args.batch_size, True,  args.seed, args.num_workers, device)
        test_loader  = make_loader(test_dataset,   args.batch_size, False, args.seed, args.num_workers, device)

        # ── Step 2: Train a fresh AE only on the 9 normal digits ──
        model = ConvAutoencoder(args.anomaly_latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.anomaly_epochs):
            train_one_epoch(model, train_loader, optimizer, device, args.anomaly_loss, args.max_train_batches)

        # ── Step 3: Score every test image ──
        errors, labels = reconstruction_errors(model, test_loader, device, args.max_test_batches)
        # Ground-truth anomaly flag: 1 for the excluded digit, 0 for the 9 normal ones.
        is_anomaly = (labels == excluded_digit).astype(np.int32)

        # ── Step 4: Metrics ──
        roc_auc = roc_auc_score(is_anomaly, errors)
        avg_precision = average_precision_score(is_anomaly, errors)

        # Operational threshold: 95th percentile of normal reconstruction errors.
        # Any test sample scoring above this is flagged as anomalous.
        threshold = np.percentile(errors[is_anomaly == 0], 95)
        predicted = errors >= threshold
        # Fraction of true anomalies that exceed the threshold.
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

        # ── Step 5: Plot score distributions ──
        plt.figure(figsize=(7, 4))
        # Normal-class histogram (9 digits)
        plt.hist(errors[is_anomaly == 0], bins=60, alpha=0.7, label="normal", density=True)
        # Anomaly-class histogram (excluded digit)
        plt.hist(errors[is_anomaly == 1], bins=60, alpha=0.7, label=f"anomaly digit {excluded_digit}", density=True)
        # Vertical dashed line at the 95th percentile of the normal distribution
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
    Run ``python cae.py --help`` to see the automatically-generated help text.
    """
    parser = argparse.ArgumentParser(description="Autoencoder experiments for MNIST/Fashion-MNIST.")

    # ── Dataset selection ──
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST",
                        help="Which torchvision dataset to use (default: MNIST).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory to cache downloaded raw MNIST files (default: ./data).")
    parser.add_argument("--out-dir",  type=Path, default=Path("runs"),
                        help="Root directory where run sub-folders and PNGs / CSVs are saved (default: ./runs).")

    # ── Reconstruction-experiment training hyper-parameters ──
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per (loss, latent_dim) configuration (default: 10).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="SGD mini-batch size (default: 128).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[2, 8, 32, 128],
                        help="List of bottleneck sizes to sweep (default: 2 8 32 128).")
    parser.add_argument("--losses", choices=["bce", "mse"], nargs="+", default=["bce", "mse"],
                        help="Reconstruction loss functions to compare (default: bce mse).")

    # ── Infrastructure / reproducibility ──
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for deterministic results (default: 42).")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker sub-processes (default: 2).  Set to 0 if debugging.")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Cap the number of training batches per epoch (None = all batches).")
    parser.add_argument("--max-test-batches", type=int, default=None,
                        help="Cap the number of test batches during evaluation (None = all batches).")

    # ── Anomaly-detection sub-experiment ──
    parser.add_argument("--skip-anomaly", action="store_true",
                        help="If present, skip the anomaly-detection experiment entirely.")
    parser.add_argument("--anomaly-digits", type=int, nargs="+", default=list(range(10)),
                        help="Digits to try as anomalies, one-at-a-time (default: 0 1 2 ... 9).")
    parser.add_argument("--anomaly-loss", choices=["bce", "mse"], default="bce",
                        help="Loss used to train the anomaly AEs (default: bce).")
    parser.add_argument("--anomaly-latent-dim", type=int, default=32,
                        help="Bottleneck size for the anomaly AEs (default: 32).  "
                             "Can differ from the reconstruction sweep.")
    parser.add_argument("--anomaly-epochs", type=int, default=5,
                        help="Training epochs for each anomaly AE (default: 5). "
                             "Usually shorter than the main run because we only care about ranking.")

    return parser.parse_args()


def main():
    """
    Main execution pipeline:

      1. Parse CLI flags.
      2. Fix RNG seeds and pick the best available device.
      3. Create a unique run directory inside ``--out-dir``.
      4. Load train / test datasets and build DataLoaders.
      5. For every combination of (loss, latent_dim):
            - Train the AE for N epochs.
            - Log epoch losses.
            - Save model checkpoint, reconstruction grid, and (if latent_dim==2)
              2-D latent scatter + interpolation strip.
      6. Aggregate all logs into a CSV and a comparison plot.
      7. If --skip-anomaly is not set, run the one-class anomaly experiment
         for each digit defined by --anomaly-digits.
    """
    # ── 1. Parse args ──
    args = parse_args()

    # ── 2. Reproducibility + device ──
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ── 3. Create run directory (e.g. ``runs/ae_mnist_seed42/``) ──
    run_dir = args.out_dir / f"ae_{args.dataset.lower()}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Data ──
    train_dataset = get_dataset(args.dataset, args.data_dir, train=True)
    test_dataset  = get_dataset(args.dataset, args.data_dir, train=False)
    train_loader = make_loader(train_dataset, args.batch_size, True,  args.seed, args.num_workers, device)
    test_loader  = make_loader(test_dataset,  args.batch_size, False, args.seed, args.num_workers, device)

    # ── 5. Reconstruction sweep ──
    all_rows = []
    for loss_name in args.losses:
        for latent_dim in args.latent_dims:
            _, rows = train_ae_for_latent_dim(
                args, latent_dim, loss_name, train_loader, test_loader, run_dir, device
            )
            all_rows.extend(rows)

    # ── 6. Save aggregated logs and comparison plot ──
    pd.DataFrame(all_rows).to_csv(run_dir / "experiment_log.csv", index=False)
    save_loss_comparison(all_rows, run_dir)

    # ── 7. Optionally run anomaly detection ──
    if not args.skip_anomaly:
        anomaly_rows = run_anomaly_experiment(args, train_dataset, test_dataset, run_dir, device)
        pd.DataFrame(anomaly_rows).to_csv(run_dir / "anomaly_log.csv", index=False)

    print(f"Done. Results saved to: {run_dir.resolve()}")


# Python entry-point guard: only execute `main()` when the script is launched
# directly by the interpreter, NOT when it is imported as a module.
# This allows other scripts to ``import cae`` and reuse the classes/functions.
if __name__ == "__main__":
    main()
