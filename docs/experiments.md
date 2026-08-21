# VAE Experiment Design

Required experiments from the course assignment (Topic 9, VAE section). Results will be compared with the teammate working on AE.

---

## 1. Latent Dimension Study

### Goal

Evaluate the effect of `latent_dim` ∈ {2, 8, 32, 128} on reconstruction quality.

### Design

```python
latent_dims = [2, 8, 32, 128]
results = {}

for dim in latent_dims:
    model = VAE(latent_dim=dim, image_channels=1, image_size=28)
    history = train_vae(model, train_loader, num_epochs=50)
    results[dim] = history["val_recon_loss"][-1]  # final reconstruction loss
```

### Expected Results

| latent_dim | Val Recon Loss | Observation |
|---|---|---|
| 2 | ~0.XX | Bottleneck too small, information loss |
| 8 | ~0.XX | Balance between compression and reconstruction |
| 32 | ~0.XX | Good reconstruction, latent space still compact |
| 128 | ~0.XX | Near-perfect reconstruction, but marginal gain over 32 |

**Pattern**: Reconstruction loss decreases with `latent_dim`, but with diminishing returns. For MNIST, `latent_dim = 32` is usually the sweet spot.

> 💡 **Assignment requirement**: with `latent_dim = 2`, draw a 2D latent space map (scatter plot of test set encodings, colored by label).

---

## 2. 2D Latent Space Visualization

### Goal

Visualize how VAE arranges different digit classes in a 2D `z ∈ ℝ²` space.

### Procedure

```python
from it4653.utils.visualization import plot_latent_space_2d

model = VAE(latent_dim=2).to(device)
train_vae(model, train_loader, num_epochs=50)

fig = plot_latent_space_2d(
    encoder=model,
    dataloader=test_loader,
    device=device,
    save_path="outputs/plots/vae_latent_2d.png"
)
```

### Decoding the 2D Space

For `latent_dim = 2`, the encoder outputs `μ = (μ₁, μ₂)` and `logvar = (log σ₁², log σ₂²)`. For visualization, use `μ` (the deterministic part) — do not sample from `μ + σ·ε` to get a stable scatter plot.

### Expected Results

- Different digits form **separate clusters** in the 2D space.
- Clusters **do not completely overlap** — however, some easily confused digits (e.g., 3 and 8, 1 and 7) may be close together.
- Compared to AE: the VAE latent space tends to have **more uniform density** and **better continuity** (due to KL forcing toward `N(0,I)`).

---

## 3. Linear Interpolation in Latent Space

### Goal

Demonstrate that VAE produces a **more continuous latent space** than AE.

### Design

```python
from it4653.utils.visualization import plot_interpolation_grid

# Pick two images from the test set
x1 = test_dataset[0][0]   # image A
x2 = test_dataset[1][0]   # image B

# Interpolate in VAE latent space
fig = plot_interpolation_grid(
    model=vae_model,
    x1=x1,
    x2=x2,
    num_steps=10,
    save_path="outputs/plots/vae_interpolation.png"
)
```

### Interpolation Procedure

1. Encode `x₁ → z₁` and `x₂ → z₂` (use `μ`, do **not** random sample)
2. Interpolate: `z(t) = (1−t)·z₁ + t·z₂`, `t ∈ {0, 0.1, ..., 1.0}`
3. Decode each `z(t)` to get the intermediate image

### Expected Results

- **VAE**: intermediate images are smooth and meaningful. Example: digit 3 → digit 8 through plausible intermediate shapes.
- **AE (teammate)**: may produce distorted, meaningless images in between (crossing "holes" in the latent space).

> 💡 **Assignment requirement**: place AE and VAE results side by side to show VAE produces a more continuous latent space. You will need images from your teammate's AE to compare.

---

## 4. Conditional VAE — Label-Conditioned Generation

### Goal

Generate images of a desired digit by conditioning VAE on the label.

### How It Works

CVAE extends VAE by adding label `y` to both encoder and decoder:

```
x --[Encoder + y]--> μ, logvar --[Reparameterize]--> z
z + y --[Decoder + y]--> x̂
```

The label `y` is usually encoded as a **one-hot vector** or **embedding vector** before being concatenated with `x` (encoder) or `z` (decoder).

### Conditional Generation

After training CVAE:

```python
# Generate 10 images of digit 7
desired_label = 7
one_hot = F.one_hot(torch.tensor(desired_label), num_classes=10).float()

z = torch.randn(10, latent_dim)  # sample from prior N(0,I)
generated = cvae.decode(z, one_hot)  # decoder receives both z and label
```

### Expected Results

- Decoder receiving random `z` + `y = 7` produces various styles of digit 7.
- The latent space `z` is **shared** across all classes, but the decoder "reads" `z` in the context of `y`.
- Each `y` creates a **submanifold** within the `z` space.

---

## Delivery Checklist (VAE Section)

- [ ] VAE code (`src/it4653/models/vae.py`) + loss function (`src/it4653/losses/vae_loss.py`)
- [ ] Reparameterization trick explained in code comments
- [ ] CVAE code (`src/it4653/models/cvae.py`)
- [ ] VAE trained with `latent_dim` ∈ {2, 8, 32, 128}
- [ ] 2D latent space map (`latent_dim = 2`)
- [ ] Linear interpolation images (compare with teammate)
- [ ] CVAE label-conditioned generation images
- [ ] Loss curves (TensorBoard or matplotlib)
- [ ] Table comparing reconstruction quality across `latent_dim`
