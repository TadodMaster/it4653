import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # Use the posterior mean at eval time for deterministic outputs.
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 64, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataset(name: str, root: Path, train: bool):
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
    for batch_id, batch in enumerate(loader):
        if max_batches is not None and batch_id >= max_batches:
            break
        yield batch


def recon_loss(x_hat: torch.Tensor, x: torch.Tensor, loss_name: str) -> torch.Tensor:
    # Sum over pixels per image, then average over the batch for a stable scale.
    if loss_name == "bce":
        return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    if loss_name == "mse":
        return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    raise ValueError(f"Unsupported reconstruction loss: {loss_name}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(q(z|x) || N(0, I)), summed over latent dims, averaged over the batch.
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size(0)


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    loss_name: str,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = recon_loss(x_hat, x, loss_name)
    kl = kl_divergence(mu, logvar)
    return recon + beta * kl, recon, kl


def train_one_epoch(model, loader, optimizer, device, loss_name: str, beta: float, max_batches=None):
    model.train()
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    seen = 0
    for x, _ in tqdm(limit_batches(loader, max_batches), desc="train", leave=False):
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        x_hat, _, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar, loss_name, beta)
        loss.backward()
        optimizer.step()
        running_total += loss.item() * x.size(0)
        running_recon += recon.item() * x.size(0)
        running_kl += kl.item() * x.size(0)
        seen += x.size(0)
    seen = max(seen, 1)
    return running_total / seen, running_recon / seen, running_kl / seen


@torch.no_grad()
def evaluate(model, loader, device, loss_name: str, beta: float, max_batches=None):
    model.eval()
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    seen = 0
    for x, _ in limit_batches(loader, max_batches):
        x = x.to(device)
        x_hat, _, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar, loss_name, beta)
        running_total += loss.item() * x.size(0)
        running_recon += recon.item() * x.size(0)
        running_kl += kl.item() * x.size(0)
        seen += x.size(0)
    seen = max(seen, 1)
    return running_total / seen, running_recon / seen, running_kl / seen


@torch.no_grad()
def save_reconstruction_grid(model, loader, device, path: Path, n: int = 16) -> None:
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    original_path = path.parent / f"{path.stem}_original{path.suffix}"
    save_image(
        x.cpu(),
        original_path,
        nrow=8,
        padding=2
    )
    x_hat, _, _, _ = model(x)
    pair_rows = torch.empty((2 * n, 1, 28, 28), device=device)
    pair_rows[0::2] = x
    pair_rows[1::2] = x_hat
    save_image(pair_rows.cpu(), path, nrow=8, padding=2)


@torch.no_grad()
def save_samples_from_prior(model, device, path: Path, n: int = 64) -> None:
    # Unique to the VAE: sample z ~ N(0, I) and decode to generate new images.
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    samples = model.decode(z)
    save_image(samples.cpu(), path, nrow=8, padding=2)


@torch.no_grad()
def plot_latent_2d(model, loader, device, path: Path, max_points: int = 5000) -> None:
    model.eval()
    zs = []
    ys = []
    count = 0
    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        zs.append(mu.cpu().numpy())
        ys.append(y.numpy())
        count += x.size(0)
        if count >= max_points:
            break
    z_all = np.concatenate(zs, axis=0)[:max_points]
    y_all = np.concatenate(ys, axis=0)[:max_points]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, s=7, cmap="tab10", alpha=0.8)
    plt.colorbar(scatter, ticks=list(range(10)))
    plt.title("VAE latent space 2D (posterior means)")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def plot_latent_manifold(model, device, path: Path, grid_size: int = 20, span: float = 3.0) -> None:
    # Only meaningful for latent_dim == 2: decode a grid over the prior.
    model.eval()
    lin = torch.linspace(-span, span, grid_size, device=device)
    # Build grid row by row (y descending, x ascending) for a natural layout.
    grid = []
    for yi in lin.flip(0):
        for xi in lin:
            grid.append(torch.stack([xi, yi]))
    z = torch.stack(grid)
    decoded = model.decode(z)
    save_image(decoded.cpu(), path, nrow=grid_size, padding=1)


@torch.no_grad()
def save_interpolation(model, dataset, device, path: Path, steps: int = 11) -> None:
    model.eval()
    first_x, first_y = dataset[0]
    second_x = None
    for x, y in dataset:
        if y != first_y:
            second_x = x
            break
    if second_x is None:
        raise RuntimeError("Could not find two samples with different labels for interpolation.")

    xa = first_x.unsqueeze(0).to(device)
    xb = second_x.unsqueeze(0).to(device)
    za, _ = model.encode(xa)
    zb, _ = model.encode(xb)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    z = (1 - alphas) * za + alphas * zb
    decoded = model.decode(z)
    save_image(decoded.cpu(), path, nrow=steps, padding=2)


def train_vae_for_latent_dim(args, latent_dim: int, loss_name: str, train_loader, test_loader, run_dir: Path, device):
    model = ConvVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_total, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, loss_name, args.beta, args.max_train_batches
        )
        test_total, test_recon, test_kl = evaluate(
            model, test_loader, device, loss_name, args.beta, args.max_test_batches
        )
        row = {
            "model": "VAE",
            "dataset": args.dataset,
            "loss_name": loss_name,
            "latent_dim": latent_dim,
            "beta": args.beta,
            "epoch": epoch,
            "train_total_loss": train_total,
            "train_recon_loss": train_recon,
            "train_kl": train_kl,
            "test_total_loss": test_total,
            "test_recon_loss": test_recon,
            "test_kl": test_kl,
            "seed": args.seed,
        }
        rows.append(row)
        print(row)

    torch.save(model.state_dict(), run_dir / f"vae_{loss_name}_latent_{latent_dim}.pt")
    save_reconstruction_grid(model, test_loader, device, run_dir / f"recon_{loss_name}_latent_{latent_dim}.png")
    save_samples_from_prior(model, device, run_dir / f"samples_{loss_name}_latent_{latent_dim}.png")
    if latent_dim == 2:
        plot_latent_2d(model, test_loader, device, run_dir / f"latent_map_2d_{loss_name}.png")
        plot_latent_manifold(model, device, run_dir / f"latent_manifold_{loss_name}.png")
        save_interpolation(model, test_loader.dataset, device, run_dir / f"interpolation_vae_{loss_name}.png")
    return model, rows


def save_loss_comparison(all_rows: list[dict], run_dir: Path) -> None:
    df = pd.DataFrame(all_rows)
    final_df = df.sort_values("epoch").groupby(["loss_name", "latent_dim"], as_index=False).tail(1)
    final_df = final_df.sort_values(["loss_name", "latent_dim"])
    final_df.to_csv(run_dir / "loss_comparison_summary.csv", index=False)

    loss_names = final_df["loss_name"].unique().tolist()
    fig, axes = plt.subplots(2, len(loss_names), figsize=(6 * len(loss_names), 8), squeeze=False)
    for col, loss_name in enumerate(loss_names):
        loss_df = final_df[final_df["loss_name"] == loss_name]
        ticks = loss_df["latent_dim"].tolist()

        ax = axes[0][col]
        ax.plot(loss_df["latent_dim"], loss_df["test_recon_loss"], marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.set_xlabel("latent dim")
        ax.set_ylabel(f"test reconstruction {loss_name.upper()}")
        ax.set_title(f"VAE trained with {loss_name.upper()}")
        ax.grid(True, alpha=0.3)

        ax = axes[1][col]
        ax.plot(loss_df["latent_dim"], loss_df["test_kl"], marker="o", color="tab:orange")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.set_xlabel("latent dim")
        ax.set_ylabel("test KL divergence")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_comparison_by_latent_dim.png", dpi=160)
    plt.close(fig)


def subset_without_digit(dataset, excluded_digit: int) -> Subset:
    indices = []
    for idx, (_, y) in enumerate(dataset):
        if int(y) != excluded_digit:
            indices.append(idx)
    return Subset(dataset, indices)


@torch.no_grad()
def anomaly_scores(model, loader, device, beta: float, max_batches=None):
    # Score = per-image negative ELBO (recon BCE + beta * KL); higher means more anomalous.
    model.eval()
    scores = []
    labels = []
    for x, y in limit_batches(loader, max_batches):
        x = x.to(device)
        x_hat, _, mu, logvar = model(x)
        recon = F.binary_cross_entropy(x_hat, x, reduction="none").flatten(1).sum(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        score = recon + beta * kl
        scores.extend(score.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.asarray(scores), np.asarray(labels)


def run_anomaly_experiment(args, train_dataset, test_dataset, run_dir: Path, device):
    anomaly_rows = []
    for excluded_digit in args.anomaly_digits:
        normal_train = subset_without_digit(train_dataset, excluded_digit)
        train_loader = make_loader(normal_train, args.batch_size, True, args.seed, args.num_workers, device)
        test_loader = make_loader(test_dataset, args.batch_size, False, args.seed, args.num_workers, device)

        model = ConvVAE(args.anomaly_latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.anomaly_epochs):
            train_one_epoch(
                model, train_loader, optimizer, device, args.anomaly_loss, args.beta, args.max_train_batches
            )

        scores, labels = anomaly_scores(model, test_loader, device, args.beta, args.max_test_batches)
        is_anomaly = (labels == excluded_digit).astype(np.int32)
        roc_auc = roc_auc_score(is_anomaly, scores)
        avg_precision = average_precision_score(is_anomaly, scores)
        threshold = np.percentile(scores[is_anomaly == 0], 95)
        predicted = scores >= threshold
        recall_at_95 = (predicted[is_anomaly == 1].mean()).item()

        row = {
            "excluded_digit": excluded_digit,
            "train_loss_name": args.anomaly_loss,
            "latent_dim": args.anomaly_latent_dim,
            "beta": args.beta,
            "epochs": args.anomaly_epochs,
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "normal_p95_threshold": threshold,
            "anomaly_recall_at_normal_p95": recall_at_95,
            "seed": args.seed,
        }
        anomaly_rows.append(row)
        print(row)

        plt.figure(figsize=(7, 4))
        plt.hist(scores[is_anomaly == 0], bins=60, alpha=0.7, label="normal", density=True)
        plt.hist(scores[is_anomaly == 1], bins=60, alpha=0.7, label=f"anomaly digit {excluded_digit}", density=True)
        plt.axvline(threshold, color="black", linestyle="--", linewidth=1, label="normal p95")
        plt.xlabel("negative ELBO score")
        plt.ylabel("density")
        plt.title(f"VAE anomaly detection, excluded digit {excluded_digit}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"anomaly_digit_{excluded_digit}.png", dpi=160)
        plt.close()

    return anomaly_rows


def parse_args():
    parser = argparse.ArgumentParser(description="VAE experiments for MNIST/Fashion-MNIST.")
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[2, 8, 32, 128])
    parser.add_argument("--losses", choices=["bce", "mse"], nargs="+", default=["bce", "mse"])
    parser.add_argument("--beta", type=float, default=1.0, help="Weight on the KL term (beta-VAE).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--skip-anomaly", action="store_true")
    parser.add_argument("--anomaly-digits", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--anomaly-loss", choices=["bce", "mse"], default="bce")
    parser.add_argument("--anomaly-latent-dim", type=int, default=32)
    parser.add_argument("--anomaly-epochs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    run_dir = args.out_dir / f"vae_{args.dataset.lower()}_beta{args.beta}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = get_dataset(args.dataset, args.data_dir, train=True)
    test_dataset = get_dataset(args.dataset, args.data_dir, train=False)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed, args.num_workers, device)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.seed, args.num_workers, device)

    all_rows = []
    for loss_name in args.losses:
        for latent_dim in args.latent_dims:
            _, rows = train_vae_for_latent_dim(args, latent_dim, loss_name, train_loader, test_loader, run_dir, device)
            all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(run_dir / "experiment_log.csv", index=False)
    save_loss_comparison(all_rows, run_dir)

    if not args.skip_anomaly:
        anomaly_rows = run_anomaly_experiment(args, train_dataset, test_dataset, run_dir, device)
        pd.DataFrame(anomaly_rows).to_csv(run_dir / "anomaly_log.csv", index=False)

    print(f"Done. Results saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()