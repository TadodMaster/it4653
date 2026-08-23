# Convolutional VAE for MNIST / Fashion-MNIST

This repository contains a single-file research script that trains convolutional variational autoencoders on MNIST or Fashion-MNIST. It sweeps over latent dimensions and reconstruction losses, logs per-epoch training and test metrics, and produces visualizations of reconstructions, generated samples, latent-space structure, and interpolation trajectories. A leave-one-digit-out anomaly-detection experiment is also included as a built-in benchmark. The script is intended for researchers and students who want a self-contained VAE baseline with minimal dependencies and clear extension points.

## Features

- Train a convolutional VAE on MNIST or Fashion-MNIST with automatic dataset download.
- Joint sweep over user-specified latent dimensions and reconstruction losses (BCE or MSE).
- Beta-VAE weighting on the KL-divergence term via the `--beta` flag.
- Per-epoch CSV logging of train / test total loss, reconstruction loss, and KL divergence.
- Side-by-side reconstruction grids comparing inputs to their decoded counterparts.
- Prior sampling to generate new images from the standard-normal latent distribution.
- 2D latent-space scatter plots and latent-manifold visualizations (generated only when `latent_dim == 2`).
- Linear interpolation in latent space between two test-set samples of different classes.
- Leave-one-digit-out anomaly detection with ROC-AUC, average precision, and recall-at-normal-p95-threshold.

## Background / method

A variational autoencoder models each datapoint as being generated from a latent code. The joint distribution factorises as $p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})$, where the prior is fixed to an isotropic Gaussian $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$. Because the true posterior $p(\mathbf{z}\mid\mathbf{x})$ is intractable, the VAE introduces an approximate posterior $q(\mathbf{z}\mid\mathbf{x})$ parameterised by a neural network (the encoder). Training maximises the evidence lower bound (ELBO):

$$\mathcal{L}(\mathbf{x}) = \underbrace{\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\left[\log p(\mathbf{x}\mid\mathbf{z})\right]}_{\text{reconstruction term}} - \underbrace{D_{KL}\bigl(q(\mathbf{z}\mid\mathbf{x}) \|\| p(\mathbf{z})\bigr)}_{\text{KL regulariser}}$$

The first term rewards faithful reconstructions; the second pulls the approximate posterior toward the prior, preventing overfitting and encouraging a well-structured latent space.

The code maps each mathematical object onto a concrete function. The encoder is `ConvVAE.encode`, which passes an image through two stride-2 convolutional blocks, flattens the $64 \times 7 \times 7$ feature map, and produces posterior parameters via `self.fc_mu` and `self.fc_logvar`. The reparameterisation trick — sampling $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ — lives in `ConvVAE.reparameterize`. Decoding is handled by `ConvVAE.decode`: a linear layer reshapes the latent vector back to $64 \times 7 \times 7$, then two transposed convolutions with a final sigmoid produce the reconstructed image.

The analytical KL is computed in `kl_divergence`:

$$D_{KL}\bigl(q(\mathbf{z}\mid\mathbf{x}) \|\| \mathcal{N}(\mathbf{0}, \mathbf{I})\bigr) = -\frac{1}{2}\sum_{i=1}^{d}\bigl(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2\bigr)$$

summed over the latent dimensions and averaged over the batch. The reconstruction term is implemented in `recon_loss`. Two likelihood assumptions are supported:

- **BCE** (`loss_name="bce"`) treats each pixel as an independent Bernoulli variable. This corresponds to a Bernoulli output distribution, appropriate because the decoder ends with a sigmoid and MNIST pixels are natively in $[0, 1]$ after `ToTensor`.
- **MSE** (`loss_name="mse"`) corresponds to a Gaussian likelihood with a fixed, shared variance. This is a common but less principled choice for binary-ish data.

Both losses return the per-pixel negative log-likelihood summed over spatial dimensions and averaged over the batch. The total objective assembled in `vae_loss` is:

$$\text{recon} + \beta \cdot D_{KL}$$

where `--beta` scales the KL term. Setting $\beta = 1$ recovers the standard VELBO. Values $\beta > 1$ (as explored by Higgins et al.) encourage greater disentanglement by penalising deviations from the prior more heavily, which can improve the interpretability of the latent dimensions at a modest cost in reconstruction fidelity.

## Repository structure

```
.
├── cvae.py                          # Main experiment script (VAE training + evaluation)
└── runs/
    └── vae_mnist_beta1.0_seed42/    # Run directory: vae_<dataset>_beta<beta>_seed<seed>
        ├── experiment_log.csv           # Per-epoch train / test metrics for all (loss, latent_dim)
        ├── loss_comparison_summary.csv  # Final-epoch metrics grouped by loss and latent_dim
        ├── loss_comparison_by_latent_dim.png  # Reconstruction loss and KL vs latent dimension
        ├── vae_bce_latent_2.pt        # Model checkpoint (BCE, latent_dim=2)
        ├── vae_mse_latent_2.pt        # Model checkpoint (MSE, latent_dim=2)
        ├── recon_bce_latent_2.png     # Side-by-side input / reconstruction grid
        ├── recon_bce_latent_2_original.png  # Original images for the reconstruction grid
        ├── samples_bce_latent_2.png   # Samples drawn from N(0, I) via the decoder
        ├── latent_map_2d_bce.png      # 2D scatter plot of posterior means coloured by label
        ├── latent_manifold_bce.png    # Decoded grid over a 2D latent prior slice
        ├── interpolation_vae_bce.png  # Linear interpolation between two encoded points
        ├── anomaly_log.csv            # Anomaly detection metrics per excluded digit
        ├── anomaly_digit_0.png        # Score histogram for digit-0-as-anomaly run
        ├── anomaly_digit_1.png
        └── ...
```

> **Note:** The run directory is automatically created at runtime by `main` using the pattern `vae_<dataset.lower()>_beta<beta>_seed<seed>`. Replace the placeholders with the actual arguments you supply.

## Installation

Requires Python >= 3.10. The script uses the following third-party packages:

- `torch`, `torchvision`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `tqdm`

Install with `pip`:

```bash
pip install torch torchvision numpy scikit-learn matplotlib pandas tqdm
```

Or create a `requirements.txt`:

```text
torch>=2.2
torchvision>=0.17
numpy>=1.24
scikit-learn>=1.3
matplotlib
pandas
tqdm
```

> **Note:** The PyTorch / CUDA wheel you need depends on your platform and GPU driver. If you are on a CPU-only machine or macOS the plain PyPI wheels are fine; otherwise use the index URL that matches your CUDA version (e.g. `https://download.pytorch.org/whl/cu124`).

The MNIST and Fashion-MNIST datasets are downloaded automatically on first run and cached in the directory specified by `--data-dir`.

## Quick start

**Smoke test** — trains two latent dimensions with BCE for two epochs, limits batches, and skips the anomaly experiment. Useful for verifying the environment. Takes ~1 minute on a laptop CPU.

```bash
python cvae.py --epochs 2 --latent-dims 2 8 --losses bce \
    --max-train-batches 10 --max-test-batches 5 --skip-anomaly
```

**Default full run** — sweeps over latent dimensions `[2, 8, 32, 128]` and both losses for 10 epochs. Produces all visualisation files. Takes ~5–10 minutes on a mid-range GPU.

```bash
python cvae.py --dataset MNIST --epochs 10 --latent-dims 2 8 32 128 --losses bce mse
```

**Fashion-MNIST with stronger regularisation** — trains on Fashion-MNIST with $\beta = 2.0$ (stronger beta-VAE regularisation). Takes ~10 minutes on GPU.

```bash
python cvae.py --dataset FashionMNIST --epochs 15 --beta 2.0 --latent-dims 2 8 32
```

**Focused anomaly detection** — trains only on `latent_dim = 2` and evaluates anomaly detection for digits 0, 1, and 2.

```bash
python cvae.py --latent-dims 2 --anomaly-epochs 5 --anomaly-digits 0 1 2
```

## Command-line reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | `str` | `MNIST` | Dataset to use. Allowed values: `MNIST`, `FashionMNIST`. |
| `--data-dir` | `Path` | `data` | Root directory for automatic dataset download and caching. |
| `--out-dir` | `Path` | `runs` | Root directory where the run-specific sub-directory is created. |
| `--epochs` | `int` | `10` | Number of training epochs for each (loss, latent_dim) combination in the sweep. |
| `--batch-size` | `int` | `128` | Number of images per training / evaluation mini-batch. |
| `--lr` | `float` | `1e-3` | Learning rate passed to `torch.optim.Adam`. |
| `--latent-dims` | `int...` | `2 8 32 128` | List of latent dimensions to sweep over. |
| `--losses` | `str...` | `bce mse` | List of reconstruction losses to evaluate. Allowed values: `bce`, `mse`. |
| `--beta` | `float` | `1.0` | Scalar weight on the KL-divergence term (beta-VAE). |
| `--seed` | `int` | `42` | Global random seed for numpy, torch, Python `random`, and the DataLoader generator. |
| `--num-workers` | `int` | `2` | Number of subprocesses used by each DataLoader. |
| `--max-train-batches` | `int` | `None` | If set, cap each training epoch to this many batches (useful for quick smoke tests). |
| `--max-test-batches` | `int` | `None` | If set, cap each test evaluation to this many batches. |
| `--skip-anomaly` | `flag` | — | Omit the leave-one-digit-out anomaly-detection experiment. |
| `--anomaly-digits` | `int...` | `0 1 2 3 4 5 6 7 8 9` | Digits to hold out as anomalies, one at a time. |
| `--anomaly-loss` | `str` | `bce` | Reconstruction loss used during anomaly-detection model training. |
| `--anomaly-latent-dim` | `int` | `32` | Latent dimension for the anomaly-detection VAE. |
| `--anomaly-epochs` | `int` | `5` | Number of training epochs for each anomaly-detection model. |

## Outputs

### `experiment_log.csv`

This file contains one row per epoch for every combination of `--losses` and `--latent-dims`. Exact columns:

| Column | Meaning |
|--------|---------|
| `model` | Always `"VAE"`. |
| `dataset` | Dataset name (`"MNIST"` or `"FashionMNIST"`). |
| `loss_name` | `bce` or `mse`. |
| `latent_dim` | Latent dimension for this training run. |
| `beta` | Beta value used. |
| `epoch` | 1-indexed epoch number. |
| `train_total_loss` | Mean per-sample `recon + beta * KL` on the training set. |
| `train_recon_loss` | Mean per-sample reconstruction loss on the training set. |
| `train_kl` | Mean per-sample KL divergence on the training set. |
| `test_total_loss` | Same metrics computed on the held-out test set. |
| `test_recon_loss` |  |
| `test_kl` |  |
| `seed` | Random seed used. |

### `loss_comparison_summary.csv`

Contains the final-epoch row for each (`loss_name`, `latent_dim`) pair, sorted by loss and latent dimension. Columns are identical to `experiment_log.csv`.

### `loss_comparison_by_latent_dim.png`

A two-row figure. The top row shows test reconstruction loss vs latent dimension for each loss (one subplot per loss); the bottom row shows test KL divergence vs latent dimension. Both axes use a base-2 logarithmic scale for the latent dimension axis.

### Model checkpoints

One checkpoint per (`loss_name`, `latent_dim`):

```
vae_{loss_name}_latent_{latent_dim}.pt
```

Reload for inference or further fine-tuning:

```python
from cvae import ConvVAE
import torch

model = ConvVAE(latent_dim=2)
model.load_state_dict(torch.load("runs/vae_mnist_beta1.0_seed42/vae_bce_latent_2.pt"))
model.eval()

# Generate 16 samples from the prior
with torch.no_grad():
    z = torch.randn(16, 2)
    samples = model.decode(z)
```

### Image files

| File pattern | What it shows |
|---|---|
| `recon_{loss}_latent_{d}.png` | Pairs of `(input, reconstruction)` tiled in rows of 8. |
| `recon_{loss}_latent_{d}_original.png` | The original inputs that correspond to the reconstruction grid. |
| `samples_{loss}_latent_{d}.png` | 64 images decoded from latent codes sampled i.i.d. from $\mathcal{N}(\mathbf{0}, \mathbf{I})$. |
| `latent_map_2d_{loss}.png` | Scatter plot of the first two posterior-mean coordinates for the first 5000 test points, coloured by digit label. Only produced when `latent_dim == 2`. |
| `latent_manifold_{loss}.png` | A $20 \times 20$ grid of decoded images spanning $[-3, 3]^2$ in latent space. Only produced when `latent_dim == 2`. |
| `interpolation_vae_{loss}.png` | Linear interpolation between the latent codes of two test-set images from different classes (11 steps). Only produced when `latent_dim == 2`. |

### `anomaly_log.csv`

One row per excluded digit in the anomaly-detection experiment. Exact columns:

| Column | Meaning |
|--------|---------|
| `excluded_digit` | The digit held out as the anomaly class. |
| `train_loss_name` | Loss used to train the anomaly VAE. |
| `latent_dim` | Latent dimension of the anomaly VAE. |
| `beta` | Beta value used during anomaly training. |
| `epochs` | Number of training epochs. |
| `roc_auc` | ROC-AUC of the anomaly score against binary anomaly labels. |
| `average_precision` | Average precision score (area under precision-recall curve). |
| `normal_p95_threshold` | 95th percentile of the score distribution on the normal class. |
| `anomaly_recall_at_normal_p95` | Fraction of true anomalies that exceed the `normal_p95_threshold`. |
| `seed` | Random seed used. |

### `anomaly_digit_{N}.png`

Density histograms of the negative-ELBO anomaly scores for the normal and anomalous test points. A vertical dashed line marks the `normal_p95_threshold`.

## Experiments

### Latent-dimension and reconstruction-loss sweep

**Question:** How does latent dimensionality and the choice of reconstruction likelihood (Bernoulli via BCE vs Gaussian via MSE) affect reconstruction fidelity and the regularisation imposed by the KL term? Is there an obvious trade-off where larger latent spaces overfit or fail to remain close to the prior?

**Setup:** The script trains a separate `ConvVAE` for every combination of `args.losses` and `args.latent_dims` for `args.epochs` epochs on the full training split. Evaluation is always performed on the untouched test split.

**Metrics:**
- `test_recon_loss` — final-epoch per-sample reconstruction error on the test set.
- `test_kl` — final-epoch per-sample KL divergence.

Both are plotted against `latent_dim` in `loss_comparison_by_latent_dim.png`.

**What to look for:** A well-behaved VAE should show decreasing reconstruction loss as latent dimension grows, but the KL term should increase (or at least not collapse to zero), indicating the posterior is still being regularised. If the KL drops to near zero, the model may be ignoring the latent code (posterior collapse), which appears as perfect reconstructions but poor sampling from the prior.

| Dataset | Loss | latent_dim | Beta | test_recon_loss | test_kl |
|---------|------|-----------|------|-----------------|---------|
| ...     | ...  | ...       | ...  | ...             | ...     |

### Leave-one-class-out anomaly detection

**Question:** Can a VAE trained only on "normal" digits detect anomalous digits based on the negative ELBO? What is the distribution of anomaly scores, and how does a simple percentile-based threshold perform?

**Setup:** For each digit in `--anomaly-digits`, the script:
1. Creates a training subset that excludes that digit (`subset_without_digit`).
2. Trains a fresh `ConvVAE(args.anomaly_latent_dim)` on the normal subset for `args.anomaly_epochs`.
3. Computes the per-image anomaly score on the full test set.

**Score definition:** The anomaly score is the per-sample negative ELBO,

$$\text{score}(\mathbf{x}) = \text{recon}_{\text{BCE}}(\mathbf{x}, \mathbf{\hat{x}}) + \beta \cdot D_{KL}\bigl(q(\mathbf{z}\mid\mathbf{x}) \|\| \mathcal{N}(\mathbf{0}, \mathbf{I})\bigr)$$

computed via `anomaly_scores`. Higher scores indicate more anomalous examples.

**Metrics:**
- `roc_auc` — `sklearn.metrics.roc_auc_score` between binary anomaly labels and the scores.
- `average_precision` — `sklearn.metrics.average_precision_score` on the same labels and scores.
- `normal_p95_threshold` — 95th percentile of the score distribution on the normal (non-anomaly) test points.
- `anomaly_recall_at_normal_p95` — `mean(scores[anomaly] >= threshold)`, i.e. the fraction of true anomalies that exceed the normal p95 threshold.

| excluded_digit | Beta | roc_auc | average_precision | anomaly_recall_at_normal_p95 |
|----------------|------|---------|-------------------|------------------------------|
| ...            | ...  | ...     | ...               | ...                          |

## Reproducibility

The `set_seed` function fixes the random state across numpy, Python `random`, and PyTorch (CPU and CUDA). It also sets:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

The DataLoader returned by `make_loader` receives a `torch.Generator` seeded with the same `args.seed`, so the batch order is fixed.

**Caveats:** Full determinism cannot be guaranteed on all hardware configurations. CUDA / MPS kernels, in particular convolution and matrix-multiplication implementations, may introduce nondeterministic floating-point round-off differences even with seeds set. DataLoader workers (`--num-workers > 0`) add a separate source of variability in how examples are batched, even when the generator is seeded. For the most reproducible results on GPU, use `--num-workers 0`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `RuntimeError: CUDA out of memory` | Batch size or latent dimension too large for GPU memory. | Reduce `--batch-size` or use `--max-train-batches`. |
| Falls back to CPU even though a GPU is present | PyTorch was installed for CPU only, or CUDA version mismatch. | Re-install the correct `torch` wheel for your CUDA driver. |
| `RuntimeError` from `DataLoader` workers on Windows/macOS | Multiprocessing issues with fork/spawn. | Set `--num-workers 0`. |
| Loss becomes `NaN` during training | Learning rate too high or numerical instability in the KL computation. | Lower `--lr` (e.g. `1e-4`) or check that inputs are in `[0, 1]`. |
| Test KL drops to ~0 while reconstruction loss is very low | Posterior collapse: the encoder predicts near-zero variance and the decoder ignores the latent code. | Increase `--beta` (e.g. `2.0–4.0`) or experiment with KL annealing. |
| Dataset download hangs or HTTP error | Transient network issue or cached partial download. | Delete `data/MNIST/raw` (or the relevant subdirectory) and retry. |

## Extending the code

**Add a new reconstruction loss:**
Edit `recon_loss` to accept a new `loss_name`, compute the per-pixel loss with `reduction="sum"`, and divide by the batch size. Add the new choice to the `choices` lists in `parse_args` for both `--losses` and `--anomaly-loss`.

**Swap the dataset:**
Modify `get_dataset`: add a new conditional branch that returns the desired `torchvision.datasets` object (e.g. `KMNIST`, `EMNIST`), or write a custom `Dataset` that yields `(image, label)` tuples with `ToTensor` applied.

**Change the encoder / decoder architecture:**
Edit `ConvVAE.__init__`. The current encoder produces a $64 \times 7 \times 7$ feature map because two stride-2 convolutions reduce a $28 \times 28$ image by factors of 2. If you change kernel sizes, strides, or number of layers, update the flatten dimension in `fc_mu`, `fc_logvar`, and `decoder_fc` accordingly, and adjust the reshape in `decode`.

**Move from VAE to conditional VAE (CVAE):**
Add a label-embedding layer to `ConvVAE.__init__` (e.g. `nn.Embedding(num_classes, embed_dim)`), then concatenate the embedded label to the flattened encoder output before `fc_mu` / `fc_logvar`, and concatenate again to the latent code before `decoder_fc`. Update `encode`, `decode`, and `forward` to accept and propagate the label tensor. Conditional generation then proceeds by fixing a class label and sampling latent codes as usual.

## References

- Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes.
- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
- Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models.
