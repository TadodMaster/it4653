# VAE / CVAE Training Guide

## Environment Setup

This project uses `uv` for environment and dependency management.

```bash
# 1. Create virtual environment
uv venv --python python3.12

# 2. Install dependencies (torch, torchvision, etc.)
uv pip install -e ".[dev]"

# Verify
uv run python -c "import torch; print(torch.__version__)"
```

## Project Structure

```
src/it4653/
├── models/          # vae.py, cvae.py  ← model implementations
├── losses/          # vae_loss.py      ← ELBO loss
├── data/            # datasets.py      # MNIST, Fashion-MNIST, CelebA loaders
├── training/        # trainers.py      # train_vae(), train_cvae()
├── evaluation/      # metrics.py       # Reconstruction quality evaluation
├── utils/           # visualization.py, checkpoints.py
└── experiments/     # latent_space.py, interpolation.py
```

## Training a Standard VAE

```bash
uv run python scripts/train_vae.py --config configs/vae.yaml
```

Or write a short script directly:

```python
import torch
from it4653.models.vae import VAE
from it4653.data.datasets import get_mnist_loaders
from it4653.training.trainers import train_vae

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
train_loader, val_loader = get_mnist_loaders(batch_size=128)

# Initialize model
model = VAE(latent_dim=32, image_channels=1, image_size=28).to(device)

# Train
history = train_vae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    lr=1e-3,
    device=device,
)
```

### Key Hyperparameters

| Parameter | Typical Value | Meaning |
|---|---|---|
| `latent_dim` | 2, 8, 32, 128 | Latent space dimension |
| `beta` | 1.0 | KL weight (1.0 = standard VAE) |
| `lr` | 1e-3 | Adam learning rate |
| `batch_size` | 128 | Batch size |
| `num_epochs` | 50 | Number of epochs (MNIST ~15 min on GPU) |

## Training Conditional VAE (CVAE)

```bash
uv run python scripts/train_cvae.py --config configs/cvae.yaml
```

CVAE differs from VAE in that:
- Encoder receives an additional `label` (one-hot or embedding)
- Decoder receives an additional `label`
- Result: can generate images conditioned on a desired label

## Monitoring with TensorBoard

```bash
tensorboard --logdir=./outputs/logs
```

Metrics to log:
- `train/total_loss` — ELBO = reconstruction + KL
- `train/recon_loss` — MSE / BCE reconstruction
- `train/kl_loss` — KL divergence
- `val/total_loss` — validation loss
- `images/reconstruction` — reconstruction grid per epoch
- `images/generated` — generated image grid from `z ~ N(0,I)` per epoch

## Checkpointing

Models are saved to `outputs/checkpoints/` every `save_every` epochs. To resume:

```python
from it4653.utils.checkpoints import load_checkpoint

checkpoint_path = "outputs/checkpoints/vae_checkpoint.pt"
epoch, history = load_checkpoint(model, optimizer, checkpoint_path)
```

## Estimated Training Time

| Dataset | latent_dim | GPU | Time |
|---|---|---|---|
| MNIST | 32 | RTX 3060 | ~2–3 min |
| MNIST | 32 | CPU | ~10–15 min |
| CelebA 64×64 | 32 | RTX 3060 | ~20–30 min (30k images) |

> ⚠️ GPU is needed for training within a reasonable time. MNIST can run on CPU but will be slow.
