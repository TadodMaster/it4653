#!/usr/bin/env python3
"""
Conditional-generation demo for the trained CVAE — "how the label INPUT controls the OUTPUT"

This script loads a checkpoint produced by ``3_conditional_variational_autoencoder.py``
and drives its decoder p(x | z, y) by hand, so the effect of the conditioning input
is visible in isolation.  Nothing is trained here.

The decoder takes TWO inputs:
    z  — the latent style code   (latent_dim numbers, sampled or encoded from an image)
    y  — the class condition     (a 10-dim one-hot vector: 3 → [0,0,0,1,0,0,0,0,0,0])
They are concatenated into a single (latent_dim + 10) vector, so "asking for a 7"
literally means putting a 1.0 in slot 7 of the second half of that vector.

Three demos:
  A) One fixed z, label swept 0..9 → the label alone decides which digit appears.
  B) A requested digit string (--digits), drawn in several independent styles →
     controllable generation: you choose the output, the model fills in the style.
  C) A real test image encoded to z, then re-decoded under every label → the
     condition overrides identity while z keeps the handwriting style.

Typical call:
    python 4_conditional_generation_demo.py --latent-dim 16 --digits 2 0 2 6
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard-library imports
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import importlib.util             # Load a module from an explicit file path
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Third-party imports
# ──────────────────────────────────────────────────────────────────────────────
import torch
from torchvision.utils import save_image


def load_module(path: Path, name: str = "cvae_model"):
    """
    Import a Python file by path.

    Needed because the model lives in ``3_conditional_variational_autoencoder.py``:
    a module name starting with a digit is not a valid Python identifier, so a plain
    ``import`` is impossible.  ``spec_from_file_location`` bypasses the name rules.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model definitions from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_checkpoint(runs_dir: Path, loss: str, latent_dim: int) -> Path:
    """
    Locate ``cvae_<loss>_latent_<dim>.pt`` anywhere under `runs_dir`.

    The training script writes it into
    ``runs/cvae_<dataset>_beta<beta>_seed<seed>/``, so a recursive glob finds it
    without the caller having to spell out the beta and the seed.
    """
    matches = sorted(runs_dir.glob(f"**/cvae_{loss}_latent_{latent_dim}.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint cvae_{loss}_latent_{latent_dim}.pt under {runs_dir}. "
            f"Train one first with 3_conditional_variational_autoencoder.py."
        )
    return matches[-1]   # Most recent run wins if several exist


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO A — one style, every label
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def demo_one_style_all_labels(cvae, model, device, out_path: Path) -> None:
    """
    Hold z constant and sweep the condition over all ten classes.

    Because only the label changes between cells, whatever varies across the row is
    attributable to the conditioning input alone.  Output: a single row of 10 images.
    """
    z = torch.randn(1, model.latent_dim, device=device)          # one fixed style
    labels = torch.arange(model.num_classes, device=device)      # the input we control
    y_onehot = cvae.one_hot_labels(labels, model.num_classes, device)
    imgs = model.decode(z.expand(model.num_classes, -1), y_onehot)
    save_image(imgs.cpu(), out_path, nrow=model.num_classes, padding=2)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO B — request a specific digit string
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def demo_requested_digits(cvae, model, device, digits: list[int], styles: int, out_path: Path) -> None:
    """
    Generate exactly the digits asked for, once per style.

    Row s uses one z drawn from the prior N(0, I) and holds it across the whole row,
    so each row is "the same handwriting" spelling the requested sequence.  This is
    the operation an unconditional VAE cannot perform: it can sample digits, but it
    cannot be told *which* digits to sample.
    """
    y_req = cvae.one_hot_labels(torch.tensor(digits), model.num_classes, device)
    rows = []
    for _ in range(styles):
        z_style = torch.randn(1, model.latent_dim, device=device).expand(len(digits), -1)
        rows.append(model.decode(z_style, y_req))
    save_image(torch.cat(rows, dim=0).cpu(), out_path, nrow=len(digits), padding=2)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO C — real image, re-decoded under every label
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def demo_real_image_relabelled(cvae, model, dataset, device, n: int, out_path: Path) -> list[int]:
    """
    Encode real test images, then decode each one under all ten conditions.

    Layout: each row is [ original | as 0 | as 1 | ... | as 9 ].
    The encoder is given the image's TRUE label (the setting it was trained on), so μ
    is the style code the decoder expects; only the decoder's condition is then
    changed.  Rows that keep their stroke style while changing identity are direct
    evidence that z carries style and y carries class.

    Returns the true labels of the sampled images, for the console log.
    """
    all_labels = torch.arange(model.num_classes, device=device)
    y_all = cvae.one_hot_labels(all_labels, model.num_classes, device)

    rows = []
    true_labels = []
    for idx in range(n):
        x_i, y_i = dataset[idx]
        x_i = x_i.unsqueeze(0).to(device)
        true_labels.append(int(y_i))
        # Style code for this image (eval mode ⇒ reparameterize returns μ).
        mu, _ = model.encode(x_i, cvae.one_hot_labels(torch.tensor([int(y_i)]), model.num_classes, device))
        decoded = model.decode(mu.expand(model.num_classes, -1), y_all)
        rows.append(torch.cat([x_i, decoded], dim=0))       # (1 + num_classes, 1, 28, 28)

    save_image(torch.cat(rows, dim=0).cpu(), out_path, nrow=1 + model.num_classes, padding=2)
    return true_labels


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Conditional-generation demo for a trained CVAE.")
    parser.add_argument("--model-file", type=Path, default=Path("3_conditional_variational_autoencoder.py"),
                        help="File defining ConvCVAE / one_hot_labels (default: 3_conditional_variational_autoencoder.py).")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint .pt to load.  Omit to auto-discover under --runs-dir.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Where to search for the checkpoint (default: ./runs).")
    parser.add_argument("--latent-dim", type=int, default=16,
                        help="Latent size of the checkpoint — must match exactly (default: 16).")
    parser.add_argument("--loss", choices=["bce", "mse"], default="bce",
                        help="Loss the checkpoint was trained with, used to build its filename (default: bce).")
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST",
                        help="Dataset supplying the real images for demo C (default: MNIST).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Raw-data cache directory (default: ./data).")
    parser.add_argument("--digits", type=int, nargs="+", default=[2, 0, 2, 6],
                        help="The digit sequence to request in demo B (default: 2 0 2 6).")
    parser.add_argument("--styles", type=int, default=3,
                        help="How many independent styles (rows) to draw in demo B (default: 3).")
    parser.add_argument("--n-real", type=int, default=4,
                        help="How many real test images to relabel in demo C (default: 4).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write the PNGs (default: next to the checkpoint).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the sampled styles (default: 0).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Model definitions are imported from the training script — no duplicated
    # architecture code, so the demo can never drift out of sync with the model.
    cvae = load_module(args.model_file)

    torch.manual_seed(args.seed)
    device = cvae.get_device()

    checkpoint = args.checkpoint or find_checkpoint(args.runs_dir, args.loss, args.latent_dim)
    out_dir = args.out_dir or checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model = cvae.ConvCVAE(args.latent_dim, num_classes=cvae.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()                     # Deterministic: reparameterize() returns μ
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")

    # Show the conditioning input itself — this IS the interface.
    example = cvae.one_hot_labels(torch.tensor([7]), cvae.NUM_CLASSES, device)
    print(f"\nAsking for a 7 -> y = {[int(v) for v in example[0].tolist()]}")
    print(f"decoder input = cat([z ({args.latent_dim} dims), y ({cvae.NUM_CLASSES} dims)]) "
          f"= {args.latent_dim + cvae.NUM_CLASSES} dims\n")

    # ── Demo A ──
    path_a = out_dir / f"demo_A_one_style_all_labels_latent_{args.latent_dim}.png"
    demo_one_style_all_labels(cvae, model, device, path_a)
    print(f"A) one z, labels 0..9              -> {path_a.name}")

    # ── Demo B ──
    path_b = out_dir / f"demo_B_requested_digits_latent_{args.latent_dim}.png"
    demo_requested_digits(cvae, model, device, args.digits, args.styles, path_b)
    print(f"B) requested {args.digits} x {args.styles} styles -> {path_b.name}")

    # ── Demo C ──
    test_dataset = cvae.get_dataset(args.dataset, args.data_dir, train=False)
    path_c = out_dir / f"demo_C_real_image_relabelled_latent_{args.latent_dim}.png"
    true_labels = demo_real_image_relabelled(cvae, model, test_dataset, device, args.n_real, path_c)
    print(f"C) real images {true_labels} re-decoded as 0..9 -> {path_c.name}")

    print(f"\nDone. Images in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
