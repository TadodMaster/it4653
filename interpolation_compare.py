#!/usr/bin/env python3
"""
So sánh nội suy tuyến tính trong không gian ẩn: Conv-AE vs Conv-VAE (latent_dim = 32)
=====================================================================================

Mục tiêu
--------
Lấy hai ảnh thật x_a, x_b (thường khác lớp), mã hoá về không gian ẩn, đi tuyến
tính giữa hai điểm ẩn:

        z(t) = (1 - t) · z_a + t · z_b ,      t = 0, …, 1

rồi giải mã từng z(t). Vẽ hai dải ảnh chồng lên nhau (AE ở trên, VAE ở dưới) để
thấy VAE cho không gian ẩn LIÊN TỤC hơn:
  • AE  : các điểm giữa thường rơi vào "vùng trống" chưa từng được huấn luyện →
          ảnh mờ, chồng bóng (ghosting), hoặc nhảy đột ngột giữa hai chữ số.
  • VAE : KL kéo hậu nghiệm về N(0, I) nên vùng giữa vẫn nằm trong miền dữ liệu →
          ảnh biến đổi mượt, mỗi bước vẫn giống một chữ số hợp lệ.

Script tái sử dụng đúng kiến trúc trong cae.py (ConvAutoencoder) và cvae.py
(ConvVAE), nạp checkpoint đã train sẵn; nếu chưa có checkpoint thì tự train
nhanh cả hai model.

Cách chạy
---------
# 1) Đã train sẵn bằng cae.py / cvae.py với --latent-dims 32
python interpolation_compare.py \
    --ae-ckpt  runs/ae_mnist_seed42/ae_bce_latent_32.pt \
    --vae-ckpt runs/vae_mnist_beta1.0_seed42/vae_bce_latent_32.pt

# 2) Chưa train gì cả -> tự train tại chỗ rồi mới nội suy
python interpolation_compare.py --train-if-missing --train-epochs 10

# 3) Chọn cặp chữ số và số bước nội suy
python interpolation_compare.py --pairs 0-1 3-8 4-9 7-2 --steps 13
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import importlib.util
import random
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")               # backend không cần màn hình (chạy trên server)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# NẠP ĐỘNG cae.py / cvae.py ĐỂ DÙNG LẠI ĐÚNG KIẾN TRÚC ĐÃ TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
# Không copy-paste lại class model: nếu kiến trúc trong cae.py/cvae.py thay đổi,
# script này vẫn khớp state_dict vì import trực tiếp từ file gốc.

def load_module(path: Path, name: str):
    """Import một file .py bất kỳ theo đường dẫn (không cần nằm trong sys.path)."""
    path = Path(path).expanduser()
    if not path.exists():
        # Thử tìm cạnh chính file script này (trường hợp chạy từ thư mục khác)
        sibling = Path(__file__).resolve().parent / path.name
        if sibling.exists():
            path = sibling
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}. Dùng --cae-file / --cvae-file để chỉ đúng đường dẫn.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════════════════════
# TIỆN ÍCH CHUNG
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Cố định mọi RNG để cùng seed thì ra cùng hình."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(root: Path, prefix: str, loss_name: str, latent_dim: int) -> Path | None:
    """
    Tự dò file checkpoint theo đúng quy ước đặt tên của cae.py / cvae.py,
    ví dụ ``ae_bce_latent_32.pt`` / ``vae_bce_latent_32.pt``.

    Thứ tự tìm kiếm:
      1. Thư mục chứa chính file interpolation_compare.py  (đặt .pt cạnh script)
      2. Thư mục làm việc hiện tại
      3. Đệ quy trong --runs-dir (nơi cae.py / cvae.py lưu mặc định), ví dụ
         ``runs/ae_mnist_seed42/ae_bce_latent_32.pt``
    """
    filename = f"{prefix}_{loss_name}_latent_{latent_dim}.pt"

    # 1 & 2: tìm phẳng cạnh script và trong cwd
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        candidate = folder / filename
        if candidate.exists():
            return candidate

    # 3: tìm đệ quy trong thư mục runs
    if root.exists():
        matches = sorted(root.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    return None


def pick_image_by_label(dataset, label: int, index_within_class: int = 0) -> torch.Tensor:
    """
    Lấy ảnh thứ `index_within_class` thuộc lớp `label` trong dataset.
    Trả về tensor shape (1, 28, 28), giá trị [0, 1].
    """
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        idxs = (targets == label).nonzero(as_tuple=True)[0].tolist()
    else:
        idxs = [i for i, t in enumerate(targets) if int(t) == label]
    if not idxs:
        raise ValueError(f"Dataset không có mẫu nào thuộc lớp {label}.")
    idx = idxs[index_within_class % len(idxs)]
    x, _ = dataset[idx]
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# LÕI: NỘI SUY TUYẾN TÍNH TRONG KHÔNG GIAN ẨN
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def interpolate_ae(model, x_a: torch.Tensor, x_b: torch.Tensor, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    AE: điểm ẩn là vector xác định z = encoder(x). Nội suy thẳng trên z.

    Returns
    -------
    frames : (steps, 28, 28) ảnh giải mã
    zs     : (steps, latent_dim) các điểm ẩn trên đoạn thẳng
    """
    model.eval()
    z_a = model.encode(x_a.unsqueeze(0))          # (1, D)
    z_b = model.encode(x_b.unsqueeze(0))          # (1, D)
    ts = torch.linspace(0.0, 1.0, steps, device=z_a.device).unsqueeze(1)   # (steps, 1)
    zs = (1.0 - ts) * z_a + ts * z_b              # broadcast → (steps, D)
    frames = model.decode(zs)                     # (steps, 1, 28, 28)
    return frames.squeeze(1).cpu().numpy(), zs.cpu().numpy()


@torch.no_grad()
def interpolate_vae(model, x_a: torch.Tensor, x_b: torch.Tensor, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    VAE: encode trả về (mu, logvar). Dùng mu — kỳ vọng của hậu nghiệm q(z|x) —
    làm điểm đại diện, KHÔNG lấy mẫu, để dải nội suy tất định và so sánh công bằng
    với AE (đúng như save_interpolation trong cvae.py).
    """
    model.eval()
    mu_a, _ = model.encode(x_a.unsqueeze(0))
    mu_b, _ = model.encode(x_b.unsqueeze(0))
    ts = torch.linspace(0.0, 1.0, steps, device=mu_a.device).unsqueeze(1)
    zs = (1.0 - ts) * mu_a + ts * mu_b
    frames = model.decode(zs)
    return frames.squeeze(1).cpu().numpy(), zs.cpu().numpy()


def path_metrics(frames: np.ndarray) -> dict:
    """
    Định lượng "độ liên tục" của dải nội suy bằng khoảng cách pixel giữa hai
    khung liên tiếp:  d_k = ||f_{k+1} - f_k||_2

    • total_length : tổng quãng đường trong không gian ảnh. Ngắn hơn ⇒ đường đi
                     thẳng thớm hơn, ít vòng vèo qua vùng trống.
    • jerkiness    : độ lệch chuẩn / trung bình của d_k (hệ số biến thiên).
                     Nhỏ ⇒ mỗi bước thay đổi đều nhau ⇒ mượt.
                     Lớn ⇒ có bước "nhảy" đột ngột (đặc trưng của AE).
    • max_step     : bước nhảy lớn nhất.
    """
    flat = frames.reshape(len(frames), -1)
    deltas = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    mean = float(deltas.mean()) if len(deltas) else 0.0
    return {
        "total_length": float(deltas.sum()),
        "mean_step": mean,
        "max_step": float(deltas.max()) if len(deltas) else 0.0,
        "jerkiness_cv": float(deltas.std() / mean) if mean > 0 else 0.0,
        "deltas": deltas,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VẼ HÌNH
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pair_comparison(
    ae_frames: np.ndarray,
    vae_frames: np.ndarray,
    label_a: int,
    label_b: int,
    latent_dim: int,
    out_path: Path,
) -> None:
    """
    Một cặp ảnh → một hình gồm 2 hàng đặt cạnh (chồng) nhau:
        hàng 1: AE  (nội suy trên z)
        hàng 2: VAE (nội suy trên mu)
    Hai đầu mút của mỗi hàng chính là ảnh tái tạo của x_a và x_b.
    """
    steps = ae_frames.shape[0]
    fig, axes = plt.subplots(2, steps, figsize=(steps * 0.85, 2.6),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    ts = np.linspace(0, 1, steps)

    for row, (frames, name) in enumerate([(ae_frames, "AE"), (vae_frames, "VAE")]):
        for col in range(steps):
            ax = axes[row, col]
            ax.imshow(frames[col], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if row == 0:
                ax.set_title(f"t={ts[col]:.2f}", fontsize=6, pad=2)
        # Nhãn hàng nằm bên trái
        axes[row, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=18,
                                va="center", fontweight="bold")

    fig.suptitle(
        f"Nội suy tuyến tính trong không gian ẩn (latent_dim={latent_dim}): "
        f"{label_a} → {label_b}\n"
        "AE: điểm giữa dễ mờ/chồng bóng  |  VAE: chuyển tiếp mượt, luôn nằm trong miền dữ liệu",
        fontsize=min(9.0, 1.15 * steps),
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.02)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grid_all_pairs(results: list[dict], latent_dim: int, out_path: Path) -> None:
    """
    Hình tổng: mỗi cặp chiếm 2 hàng (AE trên, VAE dưới), tất cả các cặp xếp dọc
    trong cùng một figure — đây là hình chính để đưa vào báo cáo.
    """
    n_pairs = len(results)
    steps = results[0]["ae_frames"].shape[0]
    fig, axes = plt.subplots(2 * n_pairs, steps,
                             figsize=(steps * 0.8, 2 * n_pairs * 0.95),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.08})

    for p, res in enumerate(results):
        for sub, key, name in [(0, "ae_frames", "AE"), (1, "vae_frames", "VAE")]:
            row = 2 * p + sub
            for col in range(steps):
                ax = axes[row][col]
                ax.imshow(res[key][col], cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
            axes[row][0].set_ylabel(f"{name}\n{res['label_a']}→{res['label_b']}",
                                    fontsize=8, rotation=0, labelpad=24, va="center")

    # Cỡ chữ tiêu đề co theo bề rộng hình để không bị cắt khi steps nhỏ
    fig.suptitle(
        f"AE vs VAE — nội suy tuyến tính trong không gian ẩn (latent_dim={latent_dim})",
        fontsize=min(11.0, 1.55 * steps),
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.92, bottom=0.02)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_step_curves(results: list[dict], out_path: Path) -> None:
    """
    Đường cong "độ lớn thay đổi giữa hai khung liên tiếp".
    Đường VAE phẳng, đều  ⇒ mỗi bước t đi được một quãng như nhau ⇒ liên tục.
    Đường AE gồ ghề, có đỉnh nhọn ⇒ ảnh nhảy lớp đột ngột ở một điểm nào đó.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.2), squeeze=False)
    for i, res in enumerate(results):
        ax = axes[0][i]
        xs = np.arange(1, len(res["ae_metrics"]["deltas"]) + 1)
        ax.plot(xs, res["ae_metrics"]["deltas"], marker="o", ms=3, label="AE")
        ax.plot(xs, res["vae_metrics"]["deltas"], marker="s", ms=3, label="VAE")
        ax.set_title(f"{res['label_a']} → {res['label_b']}", fontsize=10)
        ax.set_xlabel("bước nội suy k")
        if i == 0:
            ax.set_ylabel(r"$\|f_{k+1}-f_k\|_2$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Thay đổi pixel giữa hai khung liên tiếp — đều & thấp ⇒ không gian ẩn liên tục hơn",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN NHANH (chỉ khi thiếu checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════

def quick_train_ae(cae, args, train_loader, device, ckpt_path: Path):
    model = cae.ConvAutoencoder(args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.train_epochs + 1):
        loss = cae.train_one_epoch(model, train_loader, opt, device, args.loss, args.max_train_batches)
        print(f"[AE ] epoch {epoch}/{args.train_epochs}  train_loss={loss:.4f}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    return model


def quick_train_vae(cvae, args, train_loader, device, ckpt_path: Path):
    model = cvae.ConvVAE(args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.train_epochs + 1):
        total, recon, kl = cvae.train_one_epoch(
            model, train_loader, opt, device, args.loss, args.beta, args.max_train_batches
        )
        print(f"[VAE] epoch {epoch}/{args.train_epochs}  total={total:.4f}  recon={recon:.4f}  kl={kl:.4f}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="So sánh nội suy tuyến tính trong không gian ẩn của AE và VAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Nguồn model
    p.add_argument("--cae-file", type=Path, default=Path("cae.py"), help="Đường dẫn file cae.py (định nghĩa ConvAutoencoder).")
    p.add_argument("--cvae-file", type=Path, default=Path("cvae.py"), help="Đường dẫn file cvae.py (định nghĩa ConvVAE).")
    p.add_argument("--ae-ckpt", type=Path, default=None, help="Checkpoint AE .pt. Bỏ trống = tự dò trong --runs-dir.")
    p.add_argument("--vae-ckpt", type=Path, default=None, help="Checkpoint VAE .pt. Bỏ trống = tự dò trong --runs-dir.")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Thư mục chứa các run của cae.py / cvae.py.")

    # Cấu hình model
    p.add_argument("--latent-dim", type=int, default=32, help="Kích thước không gian ẩn (phải khớp checkpoint).")
    p.add_argument("--loss", choices=["bce", "mse"], default="bce", help="Loss dùng khi train (để tìm đúng tên checkpoint).")
    p.add_argument("--beta", type=float, default=1.0, help="Beta của VAE (chỉ dùng nếu phải train lại).")

    # Dữ liệu
    p.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    # Nội suy
    p.add_argument("--pairs", nargs="+", default=["0-1", "3-8", "4-9", "7-2"],
                   help="Các cặp lớp cần nội suy, dạng A-B (ví dụ 3-8).")
    p.add_argument("--steps", type=int, default=11, help="Số điểm t trên đoạn thẳng, gồm cả hai đầu mút.")
    p.add_argument("--sample-index", type=int, default=0, help="Lấy ảnh thứ mấy trong mỗi lớp (đổi số này để có cặp khác).")

    # Train dự phòng
    p.add_argument("--train-if-missing", action="store_true", help="Nếu không tìm thấy checkpoint thì train tại chỗ.")
    p.add_argument("--train-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train-batches", type=int, default=None)

    # Đầu ra
    p.add_argument("--out-dir", type=Path, default=Path("runs/interp_ae_vs_vae"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Nạp định nghĩa model từ hai file gốc ──
    cae = load_module(args.cae_file, "cae_mod")
    cvae = load_module(args.cvae_file, "cvae_mod")

    # ── Dữ liệu: dùng tập test để chọn ảnh nội suy ──
    test_dataset = cae.get_dataset(args.dataset, args.data_dir, train=False)

    # ── Tìm / nạp checkpoint ──
    ae_ckpt = args.ae_ckpt or find_checkpoint(args.runs_dir, "ae", args.loss, args.latent_dim)
    vae_ckpt = args.vae_ckpt or find_checkpoint(args.runs_dir, "vae", args.loss, args.latent_dim)

    ae_model = cae.ConvAutoencoder(args.latent_dim).to(device)
    vae_model = cvae.ConvVAE(args.latent_dim).to(device)

    need_train = (ae_ckpt is None or not Path(ae_ckpt).exists()
                  or vae_ckpt is None or not Path(vae_ckpt).exists())

    if need_train:
        if not args.train_if_missing:
            raise SystemExit(
                "Không tìm thấy checkpoint.\n"
                f"  AE : {ae_ckpt}\n  VAE: {vae_ckpt}\n"
                "→ Chỉ đường dẫn bằng --ae-ckpt/--vae-ckpt, hoặc thêm --train-if-missing để train tại chỗ.\n"
                f"Gợi ý train sẵn:\n"
                f"  python cae.py  --latent-dims {args.latent_dim} --losses {args.loss} --epochs 10 --skip-anomaly\n"
                f"  python cvae.py --latent-dims {args.latent_dim} --losses {args.loss} --epochs 10 --skip-anomaly"
            )
        train_dataset = cae.get_dataset(args.dataset, args.data_dir, train=True)
        train_loader = cae.make_loader(train_dataset, args.batch_size, True, args.seed, args.num_workers, device)
        if ae_ckpt is None or not Path(ae_ckpt).exists():
            ae_ckpt = args.out_dir / f"ae_{args.loss}_latent_{args.latent_dim}.pt"
            ae_model = quick_train_ae(cae, args, train_loader, device, ae_ckpt)
        if vae_ckpt is None or not Path(vae_ckpt).exists():
            vae_ckpt = args.out_dir / f"vae_{args.loss}_latent_{args.latent_dim}.pt"
            vae_model = quick_train_vae(cvae, args, train_loader, device, vae_ckpt)

    ae_model.load_state_dict(torch.load(ae_ckpt, map_location=device))
    vae_model.load_state_dict(torch.load(vae_ckpt, map_location=device))
    ae_model.eval(); vae_model.eval()
    print(f"AE  checkpoint: {ae_ckpt}")
    print(f"VAE checkpoint: {vae_ckpt}")

    # ── Chạy nội suy cho từng cặp ──
    results, metric_rows = [], []
    for pair in args.pairs:
        a_str, b_str = pair.split("-")
        label_a, label_b = int(a_str), int(b_str)

        x_a = pick_image_by_label(test_dataset, label_a, args.sample_index).to(device)
        x_b = pick_image_by_label(test_dataset, label_b, args.sample_index).to(device)

        ae_frames, _ = interpolate_ae(ae_model, x_a, x_b, args.steps)
        vae_frames, _ = interpolate_vae(vae_model, x_a, x_b, args.steps)

        ae_m = path_metrics(ae_frames)
        vae_m = path_metrics(vae_frames)

        res = dict(label_a=label_a, label_b=label_b,
                   ae_frames=ae_frames, vae_frames=vae_frames,
                   ae_metrics=ae_m, vae_metrics=vae_m)
        results.append(res)

        for name, m in [("AE", ae_m), ("VAE", vae_m)]:
            metric_rows.append({
                "pair": f"{label_a}->{label_b}",
                "model": name,
                "latent_dim": args.latent_dim,
                "steps": args.steps,
                "total_path_length": m["total_length"],
                "mean_step": m["mean_step"],
                "max_step": m["max_step"],
                "jerkiness_cv": m["jerkiness_cv"],
            })

        plot_pair_comparison(ae_frames, vae_frames, label_a, label_b,
                             args.latent_dim, args.out_dir / f"interp_{label_a}_to_{label_b}.png")
        print(f"[{label_a}→{label_b}] AE  jerkiness={ae_m['jerkiness_cv']:.3f}  max_step={ae_m['max_step']:.2f}")
        print(f"[{label_a}→{label_b}] VAE jerkiness={vae_m['jerkiness_cv']:.3f}  max_step={vae_m['max_step']:.2f}")

    # ── Hình tổng + đường cong + bảng số ──
    plot_grid_all_pairs(results, args.latent_dim, args.out_dir / "interp_grid_ae_vs_vae.png")
    plot_step_curves(results, args.out_dir / "interp_step_deltas.png")

    df = pd.DataFrame(metric_rows)
    df.to_csv(args.out_dir / "interp_metrics.csv", index=False)

    print("\n=== Tóm tắt (trung bình trên tất cả các cặp) ===")
    print(df.groupby("model")[["total_path_length", "mean_step", "max_step", "jerkiness_cv"]].mean().round(4))
    print(
        "\nĐọc kết quả: VAE thường có jerkiness_cv và max_step NHỎ HƠN — mỗi bước t đi được\n"
        "một quãng gần bằng nhau, không có cú nhảy đột ngột — bằng chứng định lượng cho việc\n"
        "không gian ẩn của VAE liên tục hơn của AE."
    )
    print(f"\nKết quả lưu tại: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
