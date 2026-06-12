# defense + privacy-utility frontier (step 4). the defense is PixelDP-style
# gaussian noise added to the smashed data before it crosses the cut, which
# corrupts exactly the channel the step-2 inversion attack reads.
#
# for one (cut, sigma, seed):
#   1. train a split model with noise(sigma) on the smashed data
#   2. test accuracy under that noise          -> utility
#   3. train the inversion decoder against the noisy encoder, get PSNR/SSIM
#                                              -> leakage
# sweep sigma and you get the frontier. sigma=0 is the undefended reference
# point (close to the step-2 cut-2 numbers but not identical -- this script
# trains on a shorter budget: 25 epochs vs 50, decoder 25 vs 30).
#
# (DP-SGD is the other obvious defense but it noises training gradients, not
# the forward activations, so it shouldn't do much against inversion. didn't
# build it -- future work.)
#
# run one config:
#   python src/defenses/defense_frontier.py --cut_layer 2 --seed 1234 \
#       --base_channels 64 --sigma 0.5 --train_epochs 25 --decoder_epochs 25
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (seed_everything, get_device, prRed, prGreen, load_cifar10,
                   ResNet18_client_side, ResNet18_server_side, mean, std)
from attacks.decoder import build_decoder
from attacks.feature_inversion import train_decoder, evaluate, load_raw_cifar10

Q2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(Q2_DIR, "results")
DEFAULT_DATA_DIR = os.path.join(Q2_DIR, "data")


class NoisyEncoder(nn.Module):
    """client encoder + gaussian noise on the output. fresh noise every forward,
    both during training and when the attacker inverts it (like deployment)."""

    def __init__(self, client: nn.Module, sigma: float):
        super().__init__()
        self.client = client
        self.sigma = float(sigma)

    def forward(self, x):
        z = self.client(x)
        if self.sigma > 0:
            z = z + torch.randn_like(z) * self.sigma
        return z


def resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def train_defended_split(encoder, server, train_loader, device, epochs, lr):
    """co-train client+server with the noise in the loop."""
    encoder.train()
    server.train()
    opt_c = torch.optim.Adam(encoder.parameters(), lr=lr)
    opt_s = torch.optim.Adam(server.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_c.zero_grad()
            opt_s.zero_grad()
            loss = criterion(server(encoder(x)), y)   # encoder adds the noise
            loss.backward()
            opt_c.step()
            opt_s.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        prRed(f"  defended-train epoch {epoch + 1}/{epochs}  loss {running / n:.4f}")


@torch.no_grad()
def model_accuracy(encoder, server, loader, device) -> float:
    encoder.eval()
    server.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = server(encoder(x)).argmax(1)   # noise still applied (deployment)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def save_summary_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(row.keys()))
        w.writerow(list(row.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut_layer", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.5,
                   help="Gaussian noise std on the smashed data (0 = undefended)")
    p.add_argument("--train_epochs", type=int, default=25)
    p.add_argument("--decoder_epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decoder_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--width", type=int, default=128, help="Decoder channel width")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--defense", type=str, default="pixeldp", choices=["pixeldp", "none"])
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR)
    args = p.parse_args()

    sigma = 0.0 if args.defense == "none" else args.sigma
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"=== Defense frontier | cut={args.cut_layer} seed={args.seed} "
          f"sigma={sigma} device={device} ===")

    # 1. train the defended split model (normalized, augmented data)
    train_set, test_set = load_cifar10(args.data_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    client = ResNet18_client_side(cut_layer=args.cut_layer,
                                  base_channels=args.base_channels).to(device)
    server = ResNet18_server_side(cut_layer=args.cut_layer,
                                  base_channels=args.base_channels).to(device)
    encoder = NoisyEncoder(client, sigma).to(device)

    train_defended_split(encoder, server, train_loader, device,
                         args.train_epochs, args.lr)
    acc = model_accuracy(encoder, server, test_loader, device)
    prGreen(f"  defended model test accuracy: {acc:.2f}%  (sigma={sigma})")

    # 2. run the inversion attack against the frozen noisy encoder
    encoder.eval()
    for prm in encoder.parameters():
        prm.requires_grad_(False)

    raw_train, raw_test = load_raw_cifar10(args.data_dir)
    raw_train_loader = DataLoader(raw_train, batch_size=args.batch_size, shuffle=True)
    raw_test_loader = DataLoader(raw_test, batch_size=args.batch_size, shuffle=False)
    norm_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    decoder = build_decoder(args.cut_layer, args.base_channels, width=args.width).to(device)
    decoder = train_decoder(encoder, decoder, raw_train_loader, device, norm_mean,
                            norm_std, args.decoder_epochs, args.decoder_lr)
    metrics = evaluate(encoder, decoder, raw_test_loader, device, norm_mean, norm_std)

    prGreen(f"--- cut={args.cut_layer} sigma={sigma}: acc {acc:.2f}% | "
            f"PSNR {metrics['psnr']:.2f} dB | SSIM {metrics['ssim']:.4f}")

    out_path = os.path.join(
        args.output_dir,
        f"defense_{args.defense}_cut{args.cut_layer}_sigma{sigma}_seed{args.seed}.csv")
    save_summary_csv(out_path, {
        "defense": args.defense,
        "cut_layer": args.cut_layer,
        "seed": args.seed,
        "base_channels": args.base_channels,
        "sigma": sigma,
        "model_acc": round(acc, 4),
        "psnr": round(metrics["psnr"], 4),
        "ssim": round(metrics["ssim"], 4),
        "mse": round(metrics["mse"], 6),
    })
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
