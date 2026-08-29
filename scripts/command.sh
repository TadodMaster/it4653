#!/usr/bin/env bash
# Reproduce the experiments in this repo, in order.
# Run these from the repository root (paths below are relative to it).

# ──────────────────────────────────────────────────────────────────────────────
# 1) Deterministic convolutional autoencoder (reconstruction sweep + anomaly detection)
# ──────────────────────────────────────────────────────────────────────────────
uv run python 1_autoencoder.py --out-dir runs --losses bce --num-workers 4 --anomaly-loss bce

# ──────────────────────────────────────────────────────────────────────────────
# 2) Variational autoencoder / beta-VAE (adds prior sampling + latent manifold)
# ──────────────────────────────────────────────────────────────────────────────
uv run python 2_variational_autoencoder.py --out-dir runs --losses bce --num-workers 4 --anomaly-loss bce --beta 1.0

# ──────────────────────────────────────────────────────────────────────────────
# 3) Conditional VAE — full run: sweep over latent dims + leave-one-digit-out anomaly.
#    --anomaly-score-mode min-over-labels scores each test image under every trained
#    class and keeps the best fit, so no label is needed at test time.
# ──────────────────────────────────────────────────────────────────────────────
uv run python 3_conditional_variational_autoencoder.py --out-dir runs --losses bce --num-workers 4 --anomaly-loss bce --anomaly-score-mode min-over-labels

# 3b) Main CVAE run — latent 32, 10 epochs (CPU, ~4 min).  Test recon BCE 65.9, KL 23.4.
#     Sharpest reconstructions and the cleanest class-conditional sample grid.
#     Caveat measured on this run: a wide latent keeps more information in z, so
#     some class identity leaks in and label-swapping is less faithful than on the
#     latent-2 model (see 3c) — occasionally a row refuses to change digit.
#     Produces recon grid, samples_by_class and label_swap.
#     Results land in runs/cvae_mnist_beta1.0_seed42/
uv run python 3_conditional_variational_autoencoder.py --epochs 10 --latent-dims 32 --losses bce --skip-anomaly --samples-per-class 10 --num-workers 0

# 3b-bis) Beta sweep at latent 32 — fixes the class leakage noted above.
#     Each beta gets its own run dir (cvae_mnist_beta<beta>_seed42), so nothing collides.
#     Measured at epoch 10 (test recon / test KL):
#         beta 1 -> 65.9 / 23.4   sharpest recon, but z hoards class info: one
#                                 label-swap row refused to change digit
#         beta 2 -> 75.2 / 13.8   label control clean on all rows, style variety kept  <-- best trade-off
#         beta 4 -> 92.1 /  6.8   label control clean, but z is squeezed so hard that
#                                 the styles start to collapse (rows look alike)
for B in 2 4; do
  uv run python 3_conditional_variational_autoencoder.py --epochs 10 --latent-dims 32 --losses bce --skip-anomaly --samples-per-class 10 --num-workers 0 --beta $B
done

# 3c) Shorter run over the 2-D latent, for the visualisations that need latent_dim == 2:
#     latent_map_2d, z/label interpolation and the 10 per-class latent manifolds.
uv run python 3_conditional_variational_autoencoder.py --epochs 5 --latent-dims 2 --losses bce --skip-anomaly --samples-per-class 10 --num-workers 0

# ──────────────────────────────────────────────────────────────────────────────
# 4) Conditional-generation demo — shows how the label INPUT drives the OUTPUT.
#    Loads a checkpoint from step 3/3b (auto-discovered under --runs-dir) and only
#    runs the decoder; nothing is trained.  --latent-dim must match the checkpoint.
#      demo A: one fixed z, labels 0..9          -> the label alone picks the digit
#      demo B: --digits requested x --styles rows -> generate exactly what you ask for
#      demo C: real image encoded, re-decoded 0..9 -> label sets identity, z keeps style
# ──────────────────────────────────────────────────────────────────────────────
uv run python 4_conditional_generation_demo.py --latent-dim 32 --digits 2 0 2 6 --styles 3 --n-real 4

# Demo against a specific beta run — pass --checkpoint, because auto-discovery would
# otherwise pick whichever run dir sorts last.  beta 2 is the recommended one.
uv run python 4_conditional_generation_demo.py --latent-dim 32 --checkpoint runs/cvae_mnist_beta2.0_seed42/cvae_bce_latent_32.pt --digits 2 0 2 6 --styles 3 --n-real 4
uv run python 4_conditional_generation_demo.py --latent-dim 32 --checkpoint runs/cvae_mnist_beta4.0_seed42/cvae_bce_latent_32.pt --digits 2 0 2 6 --styles 3 --n-real 4

# Same demo against the smaller checkpoints, for comparison.
uv run python 4_conditional_generation_demo.py --latent-dim 2 --digits 2 0 2 6 --styles 3 --n-real 4

# Any digit sequence works, e.g. spell a date in five different hands:
# uv run python 4_conditional_generation_demo.py --latent-dim 32 --digits 3 0 0 8 2 0 2 6 --styles 5

# ──────────────────────────────────────────────────────────────────────────────
# 5) AE vs VAE interpolation comparison.
#    Reads the latent-32 checkpoints produced by steps 1 and 2 from --runs-dir;
#    --cae-file / --cvae-file point at the model definitions to import.
# ──────────────────────────────────────────────────────────────────────────────
uv run python poc_interpolation.py --cae-file 1_autoencoder.py --cvae-file 2_variational_autoencoder.py --latent-dim 32 --pairs 0-1 3-8 4-9 7-2 --steps 11
