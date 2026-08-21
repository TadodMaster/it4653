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
├── models/          # vae.py, cvae.py
├── losses/          # vae_loss.py
├── data/            # datasets.py          # MNIST, Fashion-MNIST loaders
├── training/        # trainers.py          # train_vae(), train_cvae()
├── evaluation/      # metrics.py           # Reconstruction error, ELBO, active units
├── utils/           # visualization.py, checkpoints.py, config.py
└── experiments/     # latent_space.py, interpolation.py
```

## Switching Between Datasets

All training scripts use the dataset configured in `configs/vae.yaml` or `configs/cvae.yaml`.

| Dataset | Config key | Classes | Images |
|---|---|---|---|
| MNIST | `name: "mnist"` | 10 (digits 0–9) | 60k train / 10k test |
| **Fashion-MNIST** | `name: "fashion-mnist"` | 10 (fashion items) | 60k train / 10k test |

To switch, edit the config file:

```yaml
# configs/vae.yaml  or  configs/cvae.yaml
dataset:
  name: "fashion-mnist"   # or "mnist"
  data_root: "./data"
  batch_size: 128
  image_size: 28
```

Or override at the Python level:

```python
from it4653.data.datasets import get_mnist_loaders

train_loader, test_loader = get_mnist_loaders(
    dataset="fashion-mnist",   # "mnist" or "fashion-mnist"
    batch_size=128,
    image_size=28,
)
```

> 💡 This repo currently focuses on **Fashion-MNIST** for the course assignment. MNIST is available as an alternative for quick sanity-checking.

---

## Training a Standard VAE

### From CLI (recommended)

```bash
# With default config (Fashion-MNIST, latent_dim=32)
uv run python scripts/train_vae.py --config configs/vae.yaml

# Override latent_dim for a 2D visualization run
uv run python scripts/train_vae.py --config configs/vae.yaml --latent-dim 2

# Switch to MNIST for a quick test
# (edit configs/vae.yaml -> dataset.name: "mnist" first)
```

### From Python

```python
import torch
from it4653.models.vae import VAE
from it4653.data.datasets import get_mnist_loaders
from it4653.training.trainers import train_vae

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fashion-MNIST
train_loader, val_loader = get_mnist_loaders(
    dataset="fashion-mnist",
    batch_size=128,
    image_size=28,
)

# Model
model = VAE(latent_dim=32, image_channels=1, image_size=28).to(device)

# Train
history = train_vae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    lr=1e-3,
    beta=1.0,
    device=device,
    save_dir="./outputs/checkpoints/vae",
    log_dir="./outputs/logs/vae",
)
```

### Key Hyperparameters

| Parameter | Typical Value | Meaning |
|---|---|---|
| `latent_dim` | 2, 8, 32, 128 | Latent space dimension |
| `beta` | 1.0 | KL weight (1.0 = standard VAE) |
| `lr` | 1e-3 | Adam learning rate |
| `batch_size` | 128 | Batch size |
| `num_epochs` | 50 | Number of epochs |

---

## Training Conditional VAE (CVAE)

CVAE enables label-conditioned generation: "give me a Sneaker" or "generate 16 T-shirts".

### From CLI

```bash
uv run python scripts/train_cvae.py --config configs/cvae.yaml
```

The `configs/cvae.yaml` defaults to **Fashion-MNIST** with `num_classes: 10`.

### From Python

```python
from it4653.models.cvae import CVAE
from it4653.training.trainers import train_cvae

model = CVAE(
    latent_dim=32,
    num_classes=10,        # Fashion-MNIST: 10 classes
    image_channels=1,
    image_size=28,
    label_embed_dim=8,
)

history = train_cvae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    lr=1e-3,
    beta=1.0,
    device=device,
)
```

### Generating Conditioned Samples

```python
# After training...
model.eval()

# Generate 16 Sneakers (class 7 in Fashion-MNIST)
sneakers = model.sample_class(target_class=7, num_samples=16, device="cuda")

# Generate a grid of all 10 classes
labels = torch.arange(10).repeat(2)  # 20 images, 2 per class
grid = model.sample(labels=labels, num_samples=20, device="cuda")
```

Fashion-MNIST class index reference:
| Index | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir=./outputs/logs
```

Navigate to `http://localhost:6006` to view:

| Scalar tab | What it shows |
|---|---|
| `train/loss` | per-epoch ELBO (recon + KL) |
| `train/recon` | per-epoch BCE reconstruction loss |
| `train/kl` | per-epoch KL divergence |
| `val/loss` | validation ELBO |
| `images/reconstruction` | top row = original, bottom row = VAE reconstruction |
| `images/generated` | random samples from `z ~ N(0,I)` |

---

## Running Experiments

### 1. Latent Dimension Sweep

```bash
uv run python -c "
from it4653.experiments.latent_space import sweep_latent_dims, plot_sweep_results
results = sweep_latent_dims(
    latent_dims=[2, 8, 32, 128],
    dataset='fashion-mnist',
    num_epochs=50,
    output_dir='./outputs/sweep',
)
plot_sweep_results(results)
"
```

Output: `outputs/sweep/sweep_results.json` + `sweep_comparison.png` + `latent_space_2d.png`

### 2. Latent Space Visualization

```bash
# Must train with --latent-dim 2 first
uv run python scripts/train_vae.py --config configs/vae.yaml --latent-dim 2

# Then visualize
uv run python scripts/visualize_latent_spaces.py \
    --checkpoint outputs/checkpoints/vae/vae.pt \
    --dataset fashion-mnist \
    --num-samples 5000 \
    --output-dir outputs/plots
```

Output: `latent_space.png` (scatter), `reconstructions.png`, `interpolation.png`

### 3. Interpolation Experiment

```bash
uv run python scripts/run_interpolation.py \
    --checkpoint outputs/checkpoints/vae/vae.pt \
    --dataset fashion-mnist \
    --num-pairs 5 \
    --num-steps 10 \
    --output-dir outputs/plots
```

Output: `interpolation_grid.png`

---

## Checkpointing

Models are saved to `outputs/checkpoints/` every `save_every` epochs (default 10). To resume or evaluate:

```python
from it4653.utils.checkpoints import load_checkpoint

# Resume training
epoch, history = load_checkpoint(
    "outputs/checkpoints/vae/vae.pt",
    model, optimizer, device="cuda",
)

# Or load for evaluation only
epoch, history = load_checkpoint(
    "outputs/checkpoints/vae/vae.pt",
    model, device="cuda",
)
```

---

## Estimated Training Time (Fashion-MNIST)

| latent_dim | GPU (RTX 3060) | CPU |
|---|---|---|
| 2 | ~2 min | ~5–8 min |
| 8 | ~2 min | ~5–8 min |
| 32 | ~2–3 min | ~8–12 min |
| 128 | ~3 min | ~10–15 min |

> ⚠️ GPU is strongly recommended. A full sweep of 4 latent dimensions (50 epochs each) takes ~10–12 min on GPU, ~40–60 min on CPU.
