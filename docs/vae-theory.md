# Variational Autoencoder (VAE) Theory

## 1. From Autoencoder (AE) to Variational Autoencoder (VAE)

### Standard Autoencoder

An Autoencoder (AE) is an encoder-decoder model with architecture:

```
x --[Encoder]--> z --[Decoder]--> x̂
```

The encoder `f_φ(x)` maps input `x` to a latent vector `z` that is **deterministic**. The decoder `g_θ(z)` reconstructs `x̂`. The loss is usually MSE or cross-entropy between `x` and `x̂`.

**Limitation of AE**: the latent space `z` has no probabilistic structure. Latent points are scattered with no continuity. Sampling randomly from `z` to generate new images often produces nonsensical results because the decoder only learned to reconstruct at the exact locations where the encoder "placed" the training data points.

### Variational Autoencoder

VAE solves this by **learning a probability distribution** `q_φ(z|x)` instead of a deterministic vector. The encoder no longer outputs `z` directly but outputs **two parameters** of a Gaussian distribution:

- `μ(x)` — mean
- `σ(x)` — standard deviation (or `log(σ²)` in practice)

That is: `q_φ(z|x) = N(z; μ(x), σ²(x)I)`

The approximate posterior `q_φ(z|x)` is regularized to stay close to the prior `p(z) = N(0, I)` via KL divergence.

## 2. Loss Function — Evidence Lower Bound (ELBO)

### Variational Inference Principle

We want to learn the true posterior `p(z|x)`, but computing it directly is intractable (because `p(x|z)` is complex). VAE uses an approximate distribution `q_φ(z|x)` and maximizes the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

Two components:

| Component | Meaning | Desired Direction |
|---|---|---|
| `E[log p(x\|z)]` | Reconstruction log-likelihood — measures how well the decoder reconstructs input `x` from sample `z` | **Higher is better** (positive contribution) |
| `D_KL(q(z\|x) \| p(z))` | KL divergence between approximate posterior and standard Gaussian prior `N(0,I)` | **Lower is better** (negative sign) |

Intuition: **reconstruction loss** pulls `q(z|x)` to assign probability mass where the decoder reconstructs well; **KL loss** pulls `q(z|x)` toward `N(0,I)`, creating a structured, continuous latent space.

### Closed-form KL Divergence

When both `q(z|x)` and `p(z)` are diagonal Gaussians, KL divergence has a closed form:

$$
D_{KL}(q(z|x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)
$$

In practice, the encoder outputs `logvar = log(σ²)` instead of `σ` directly (more stable during training). Then:

```python
# logvar = log(sigma^2)
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
```

### Reconstruction Loss in Practice

For grayscale MNIST images (pixels in `[0, 1]`), reconstruction loss usually uses **Binary Cross-Entropy (BCE)** or **MSE**:

```python
# BCE reconstruction (equivalent to negative log-likelihood of Bernoulli)
recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

# Or MSE
recon_loss = F.mse_loss(recon_x, x, reduction='sum')
```

Total loss:

```python
total_loss = recon_loss + beta * kl_divergence
```

`beta = 1.0` for standard VAE; `beta > 1.0` for β-VAE (stronger KL constraint, more compact latent space but poorer reconstruction).

## 3. Reparameterization Trick

### Problem: Backpropagation Cannot Flow Through Sampling

Suppose we sample `z` directly from `N(μ, σ²)`:

```python
z = torch.normal(mu, sigma)  # z ~ N(mu, sigma^2)
```

The problem: `torch.normal()` has **undefined gradients** with respect to `mu` and `sigma`. This blocks gradient flow from `z` to the encoder parameters `φ`, making it impossible to train the encoder via gradient descent.

### Solution: "Detach" Random Noise from Gradients

Instead of sampling `z` directly, we **reparameterize** the sampling as:

$$
z = \mu + \sigma \cdot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)
$$

Where:
- `μ` and `σ` are **differentiable** outputs of the encoder
- `ε` is random noise with **no gradient** (sampled from `N(0,I)`)

### Why This Works

The distribution of `z = μ + σ·ε` (with `ε ~ N(0,I)`) is exactly `N(μ, σ²)`. But now:
- `z` is a **differentiable function** of `μ` and `σ` (addition and multiplication are differentiable)
- The random noise `ε` sits **outside the computation graph** (detached)
- Gradients from `z` flow backward through `μ` and `σ` normally

```python
def reparameterize(self, mu, logvar):
    """
    Reparameterization trick.
    logvar = log(sigma^2)
    """
    std = torch.exp(0.5 * logvar)      # sigma
    eps = torch.randn_like(std)        # epsilon ~ N(0, I), no gradient
    z = mu + std * eps                 # z ~ N(mu, sigma^2), differentiable wrt mu, std
    return z
```

**Visual illustration:**

```
AE:   x → [Encoder] → z (deterministic) → [Decoder] → x̂
                         ↑
                    gradient OK (straightforward)

VAE (without trick):
  x → [Encoder] → μ, σ → N(μ,σ²) ──► z (sampled) → [Decoder] → x̂
                              ↑               ↑
                         differentiable    no gradient!

VAE (with trick):
  x → [Encoder] → μ, σ ──► μ + σ·ε → z → [Decoder] → x̂
                              ↑  ↑
                         differentiable   ε ~ N(0,I) (no gradient)
                         ─────────────────
                              ↑
                         gradient flows smoothly
```

## 4. AE vs VAE — Why VAE Produces a More Continuous Latent Space

| Feature | Autoencoder (AE) | Variational Autoencoder (VAE) |
|---|---|---|
| Latent `z` | Deterministic | Probabilistic: `N(μ, σ²)` |
| Encoder output | `z` directly | `μ, log(σ²)` |
| Sampling from latent | Undefined (may produce weird images) | Can sample `z ~ N(0,I)`, decoder produces plausible images |
| Latent space | Scattered (sparse), with "holes" | Continuous and structured, KL forces toward `N(0,I)` |
| Linear interpolation | Often non-smooth (crosses low-density regions) | Smoother (space has structure) |
| New image generation | Poor | Good (sample from prior) |
| Loss function | MSE / BCE reconstruction | ELBO = Reconstruction − KL |

### Interpolation Experiment

Two points `z_A` and `z_B` in latent space. Linear interpolation:

```
z(t) = (1-t)·z_A + t·z_B    where t ∈ [0, 1]
```

- **AE**: `z(t)` may pass through regions where the decoder was not trained → intermediate images distorted and meaningless.
- **VAE**: `z(t)` typically lies within the high-probability region of `p(z) = N(0,I)` (because KL forces `q(z\|x)` near the origin) → decoder produces smooth, meaningful images at all intermediate points.

This interpolation experiment is one of the required deliverables for the course project.
