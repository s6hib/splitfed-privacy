# feature-inversion attack (step 2). honest-but-curious server: it follows the
# protocol but trains a decoder to reconstruct the client's input image from
# the smashed data it receives. encoder is frozen (already trained), decoder
# learns to undo it, PSNR/SSIM say how well. run this across cuts 1-4 to see
# how leakage changes with split depth.
#
# run one cut:
#   python src/attacks/feature_inversion.py --cut_layer 2 --seed 1234 \
#       --base_channels 64 --epochs 30
#
# needs a client checkpoint from a baseline run with --save_client_ckpt
# (looks in results/checkpoints/ by default). --random_encoder skips that and
# inverts an untrained encoder instead.
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import seed_everything, get_device, ResNet18_client_side, mean, std
from attacks.decoder import build_decoder, smashed_shape

Q2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(Q2_DIR, "results")
DEFAULT_FIGURES_DIR = os.path.join(Q2_DIR, "figures")
DEFAULT_CKPT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "checkpoints")
DEFAULT_DATA_DIR = os.path.join(Q2_DIR, "data")


# CIFAR-10 in [0,1] with no augmentation -- the decoder target has to be the
# exact image the encoder saw. normalization happens on the fly in the loops.
def load_raw_cifar10(data_dir):
    to_tensor = transforms.ToTensor()  # -> [0, 1], shape (3, 32, 32)
    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=to_tensor)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=to_tensor)
    return train, test


def build_encoder(cut_layer, base_channels, random_encoder, ckpt_path, device):
    """client-side encoder, with trained weights unless --random_encoder."""
    encoder = ResNet18_client_side(cut_layer=cut_layer, base_channels=base_channels)

    if not random_encoder:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No encoder checkpoint at {ckpt_path}. Run a split baseline with "
                f"--save_client_ckpt first, or pass --random_encoder."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if ckpt["cut_layer"] != cut_layer or ckpt["base_channels"] != base_channels:
            raise ValueError(
                f"Checkpoint mismatch: ckpt is cut={ckpt['cut_layer']} "
                f"base_channels={ckpt['base_channels']}, but attack wants "
                f"cut={cut_layer} base_channels={base_channels}."
            )
        encoder.load_state_dict(ckpt["state_dict"])

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder.to(device)


def train_decoder(encoder, decoder, train_loader, device, norm_mean, norm_std,
                  epochs, lr):
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    for epoch in range(epochs):
        running, n = 0.0, 0
        for images, _ in train_loader:
            x01 = images.to(device)
            x_norm = (x01 - norm_mean) / norm_std
            with torch.no_grad():
                smashed = encoder(x_norm)
            pred = decoder(smashed)
            loss = F.mse_loss(pred, x01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * x01.size(0)
            n += x01.size(0)
        print(f"  decoder epoch {epoch + 1}/{epochs}  train MSE: {running / n:.5f}")
    return decoder


@torch.no_grad()
def evaluate(encoder, decoder, test_loader, device, norm_mean, norm_std):
    from attacks.metrics import ssim
    decoder.eval()
    tot_mse = tot_psnr = tot_ssim = 0.0
    n_imgs = 0
    for images, _ in test_loader:
        x01 = images.to(device)
        x_norm = (x01 - norm_mean) / norm_std
        pred = decoder(encoder(x_norm)).clamp(0, 1)
        # metrics on cpu (sidesteps mps op gaps, cheap at this scale)
        p, t = pred.cpu(), x01.cpu()
        # per-image PSNR then mean, so the smaller last batch doesn't skew it
        mse_per_img = ((p - t) ** 2).mean(dim=(1, 2, 3))
        tot_mse += float(mse_per_img.sum())
        tot_psnr += float((10.0 * torch.log10(1.0 / (mse_per_img + 1e-8))).sum())
        tot_ssim += ssim(p, t) * x01.size(0)
        n_imgs += x01.size(0)
    return {
        "mse": tot_mse / n_imgs,
        "psnr": tot_psnr / n_imgs,
        "ssim": tot_ssim / n_imgs,
    }


@torch.no_grad()
def save_comparison_figure(encoder, decoder, test_loader, device, norm_mean,
                           norm_std, n_show, out_path, title):
    decoder.eval()
    images, _ = next(iter(test_loader))
    images = images[:n_show].to(device)
    x_norm = (images - norm_mean) / norm_std
    recon = decoder(encoder(x_norm)).clamp(0, 1).cpu()
    originals = images.cpu()

    fig, axes = plt.subplots(2, n_show, figsize=(1.4 * n_show, 3.0))
    for i in range(n_show):
        axes[0, i].imshow(originals[i].permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("recon", rotation=0, ha="right", va="center")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(row.keys()))
        writer.writerow(list(row.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut_layer", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--method", type=str, default="sflv1",
                        help="Which baseline produced the encoder (for default ckpt path)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Explicit encoder checkpoint path (overrides the default lookup)")
    parser.add_argument("--random_encoder", action="store_true",
                        help="Invert an untrained encoder (reference baseline, not a guaranteed upper bound)")
    parser.add_argument("--epochs", type=int, default=30, help="Decoder training epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128, help="Decoder channel width")
    parser.add_argument("--n_show", type=int, default=8, help="Images in the comparison figure")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures_dir", type=str, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()

    encoder_tag = "random" if args.random_encoder else "trained"
    ch, hh, ww = smashed_shape(args.cut_layer, args.base_channels)
    print(f"=== Feature inversion | cut={args.cut_layer} seed={args.seed} "
          f"encoder={encoder_tag} | smashed shape ({ch},{hh},{ww}) ===")

    ckpt_path = args.ckpt or os.path.join(
        args.ckpt_dir, f"{args.method}_client_cut{args.cut_layer}_seed{args.seed}.pt")
    encoder = build_encoder(args.cut_layer, args.base_channels, args.random_encoder,
                            ckpt_path, device)

    train_set, test_set = load_raw_cifar10(args.data_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    norm_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    decoder = build_decoder(args.cut_layer, args.base_channels, width=args.width).to(device)
    decoder = train_decoder(encoder, decoder, train_loader, device, norm_mean,
                            norm_std, args.epochs, args.lr)

    metrics = evaluate(encoder, decoder, test_loader, device, norm_mean, norm_std)
    print(f"--- cut={args.cut_layer} {encoder_tag}: "
          f"PSNR {metrics['psnr']:.2f} dB | SSIM {metrics['ssim']:.4f} | "
          f"MSE {metrics['mse']:.5f}")

    fig_path = os.path.join(
        args.figures_dir,
        f"inversion_{args.method}_{encoder_tag}_cut{args.cut_layer}_seed{args.seed}.png")
    save_comparison_figure(
        encoder, decoder, test_loader, device, norm_mean, norm_std, args.n_show,
        fig_path, f"Feature inversion — cut {args.cut_layer} ({encoder_tag} encoder)")
    print(f"Figure saved to {fig_path}")

    summary_path = os.path.join(
        args.output_dir,
        f"inversion_{args.method}_{encoder_tag}_cut{args.cut_layer}_seed{args.seed}.csv")
    save_summary_csv(summary_path, {
        "method": args.method,
        "encoder": encoder_tag,
        "cut_layer": args.cut_layer,
        "seed": args.seed,
        "base_channels": args.base_channels,
        "decoder_epochs": args.epochs,
        "psnr": round(metrics["psnr"], 4),
        "ssim": round(metrics["ssim"], 4),
        "mse": round(metrics["mse"], 6),
    })
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
