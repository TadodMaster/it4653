# VAE Experiment Design

Required experiments from the course assignment (Topic 9, VAE section). All experiments run on **Fashion-MNIST** (28×28 grayscale fashion items, 10 classes) by default.

---

## 1. Latent Dimension Study

### Goal

Evaluate the effect of `latent_dim` ∈ {2, 8, 32, 128} on reconstruction quality.

### Running the Experiment

**Using the CLI sweep script:**

```bash
uv run python -c "
from it4653.experiments.latent_space import sweep_latent_dims, plot_sweep_results
results = sweep_latent_dims(
    latent_dims=[2, 8, 32, 128],
    dataset='fashion-mnist',
    num_epochs=50,
    output_dir='./outputs/sweep',
)
plot_sweep_results(results, save_path='./outputs/sweep/sweep_comparison.png')
"
```

**Or programmatically:**

```python
from it4653.experiments.latent_space import sweep_latent_dims, plot_sweep_results

results = sweep_latent_dims(
    latent_dims=[2, 8, 32, 128],
    dataset="fashion-mnist",
    batch_size=128,
    num_epochs=50,
    lr=1e-3,
    beta=1.0,
    device="cuda",
    output_dir="./outputs/sweep",
)

# Plot comparison
fig = plot_sweep_results(results)
```

### What the Sweep Does

For each `latent_dim`:
1. Instantiates a fresh `VAE(latent_dim=dim)`
2. Trains for `num_epochs`
3. Evaluates on the test set:
   - Reconstruction error (BCE, MSE, MAE)
   - Full ELBO (`loss = recon + beta·KL`)
   - Number of "active" latent units (`KL_dim > 0.01`)
4. Saves a checkpoint: `outputs/sweep/vae_dim{dim}.pt`
5. For `latent_dim = 2`: auto-generates a 2D latent space scatter plot

### Expected Results (Fashion-MNIST)

| latent_dim | Recon Loss | Observation |
|---|---|---|
| 2 | Higher | Bottleneck too small, blurry reconstructions |
| 8 | Moderate | Balance — recognizable but not sharp |
| 32 | Low | Sharp reconstructions, most units active |
| 128 | Lowest | Near-perfect recon, diminishing returns vs 32 |

**Pattern**: Reconstruction loss decreases with `latent_dim`, but with diminishing returns. Active-unit count plateaus — for Fashion-MNIST, ~20–25 units are active even with `latent_dim = 128`, indicating the remaining dimensions collapse to the prior.

> 💡 **Assignment requirement**: for `latent_dim = 2`, plot a 2D latent space map (scatter of test-set `μ`, colored by label). The sweep auto-produces `outputs/sweep/latent_space_2d.png` when `latent_dim == 2` is in the list.

---

## 2. 2D Latent Space Visualization

### Goal

Visualize how VAE arranges different fashion classes in a 2D `z ∈ ℝ²` space.

### Running the Visualization

**From a trained model (latent_dim = 2):**

```bash
# Train first
uv run python scripts/train_vae.py --config configs/vae.yaml --latent-dim 2

# Then visualize
uv run python scripts/visualize_latent_spaces.py \
    --checkpoint outputs/checkpoints/vae/vae.pt \
    --dataset fashion-mnist \
    --num-samples 5000 \
    --output-dir outputs/plots
```

**Programmatically:**

```python
from it4653.utils.visualization import plot_latent_space_2d
from it4653.data.datasets import get_mnist_loaders

# Load your trained model
train_loader, test_loader = get_mnist_loaders(dataset="fashion-mnist")

fig = plot_latent_space_2d(
    model=vae_model,
    dataloader=test_loader,
    device="cuda",
    num_samples=5000,
    save_path="outputs/plots/latent_space_2d.png",
)
```

### Decoding the 2D Space

For `latent_dim = 2`, the encoder outputs `μ = (μ₁, μ₂)` and `logvar = (log σ₁², log σ₂²)`. For the scatter plot, use **μ** (the deterministic part) — do not sample from `μ + σ·ε`, since that adds noise and smears the clusters.

### Expected Results (Fashion-MNIST)

- Different fashion items form **separate clusters** in 2D.
- Visually similar classes (e.g., Pullover / Coat / Shirt) may be close together.
- Compared to a plain AE: the VAE latent space has **more uniform density** and **better continuity** (KL forces the posterior toward `N(0,I)`).
- The clusters should be more **overlapping** than MNIST digits because fashion items are inherently more ambiguous than handwritten numbers.

---

## 3. Linear Interpolation in Latent Space

### Goal

Demonstrate that VAE learns a **continuous latent manifold**.

### Running the Interpolation Experiment

```bash
uv run python scripts/run_interpolation.py \
    --checkpoint outputs/checkpoints/vae/vae.pt \
    --dataset fashion-mnist \
    --num-pairs 5 \
    --num-steps 10 \
    --output-dir outputs/plots
```

**Programmatically:**

```python
from it4653.experiments.interpolation import run_vae_interpolation

save_path = run_vae_interpolation(
    checkpoint_path="outputs/checkpoints/vae/vae.pt",
    dataset="fashion-mnist",
    num_pairs=5,
    num_steps=10,
    device="cuda",
    output_dir="outputs/plots",
)
```

### Interpolation Procedure

1. Pick two test images `x₁` and `x₂`
2. Encode each to `μ`: `z₁ = encode(x₁)`, `z₂ = encode(x₂)` (use `μ`, **do not** random sample)
3. Linearly interpolate: `z(t) = (1−t)·z₁ + t·z₂` for `t ∈ {0, 0.1, ..., 1.0}`
4. Decode each `z(t)` back to image space

### Expected Results

- **VAE**: intermediate images are smooth and visually plausible. Example: a T-shirt morphing into a dress through intermediate shapes that look like shirts or short dresses.
- **AE (teammate comparison)**: may produce distorted or meaningless images in between (crossing "holes" in the latent space where no training data exists).

> 💡 **Assignment requirement**: place AE and VAE results side by side. You will need images from your teammate's AE model for the comparison.

---

## 4. Conditional VAE — Label-Conditioned Generation

### Goal

Generate images of a desired fashion class by conditioning CVAE on the label.

### Running CVAE Training

```bash
uv run python scripts/train_cvae.py --config configs/cvae.yaml
```

The config defaults to `dataset: "fashion-mnist"` and `num_classes: 10`.

### How It Works

CVAE extends VAE by adding label `y` to both encoder and decoder:

```
x --[Encoder + y]--> μ, logvar --[Reparameterize]--> z
z + y --[Decoder + y]--> x̂
```

The label `y` is one-hot encoded and passed through a learned linear layer (`label_embed_dim = 8`) before concatenation.

### Conditional Generation (After Training)

```python
import torch
from it4653.models.cvae import CVAE
from it4653.utils.checkpoints import load_checkpoint

# Load trained CVAE
model = CVAE(latent_dim=32, num_classes=10)
load_checkpoint("outputs/checkpoints/cvae/cvae.pt", model, device="cuda")
model = model.to("cuda").eval()

# Generate 16 Ankle boots (class 9)
generated = model.sample_class(target_class=9, num_samples=16, device="cuda")

# Or generate a mix of classes
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5])
generated = model.sample(labels=labels, num_samples=16, device="cuda")
```

Fashion-MNIST class labels:
| Label | Class |
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

### Expected Results

- Decoder receiving random `z` + `y = 9` (Ankle boot) produces various styles of ankle boot (different angles, heel heights).
- The latent space `z` is **shared** across all 10 classes, but the decoder "reads" `z` in the context of `y`.
- Each class label creates a **submanifold** within the `z` space.

---

## Delivery Checklist (VAE Section)

- [ ] VAE code (`src/it4653/models/vae.py`) + loss (`src/it4653/losses/vae_loss.py`)
- [ ] Reparameterization trick explained in code comments
- [ ] CVAE code (`src/it4653/models/cvae.py`)
- [ ] VAE trained with `latent_dim` ∈ {2, 8, 32, 128} on **Fashion-MNIST**
- [ ] 2D latent space map (`latent_dim = 2`, Fashion-MNIST)
- [ ] Linear interpolation images (compare with teammate's AE)
- [ ] CVAE label-conditioned generation images (Fashion-MNIST)
- [ ] Loss curves (TensorBoard or matplotlib)
- [ ] Table comparing reconstruction quality across `latent_dim`
