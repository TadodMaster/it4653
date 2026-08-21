"""IT4653 — Variational Autoencoder (VAE) and Conditional VAE (CVAE).

Package structure:
    models      : VAE and CVAE architectures
    data        : Dataset loaders and transforms (MNIST, Fashion-MNIST, CelebA)
    losses      : VAE loss (reconstruction + KL divergence)
    training    : Training loops for VAE and CVAE
    evaluation  : Reconstruction quality metrics
    utils       : Visualization, checkpointing, config loading
    experiments : Latent space visualization and interpolation

For this project the teammate's responsibilities are:
    - Teammate 1 : Autoencoder (AE) models
    - This repo   : VAE / CVAE (you are here)
    - Teammate 2 : GAN / DCGAN models
"""

__version__ = "0.1.0"
