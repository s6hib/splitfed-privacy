# gradient label-leakage attack (step 3), after Li et al. ICLR 2022 ("Label
# Leakage and Protection in Two-Party Split Learning"). setup: two-party split
# learning where the labels live only with the label party. the other party
# never sees labels -- just the gradient of the loss w.r.t. the cut activation
# that comes back each step. question: how much label info is in that gradient?
#
# (worth noting the stock SplitFed code doesn't even need this attack -- clients
# literally send labels to the server in plaintext. this is the interesting
# version where the protocol actually keeps labels private.)
#
# train a plain split model at a given cut, capture per-sample gradients at the
# cut, then two attacks:
#   norm  : Li et al.'s unsupervised norm attack. on a binary task the L2 norm
#           of the cut gradient separates the classes -> leak AUC.
#   probe : supervised upper bound. logistic regression from cut gradient ->
#           class, 10-way top-1 acc (chance 10%).
#
# run one cut:
#   python src/attacks/label_leakage.py --cut_layer 2 --seed 1234 \
#       --base_channels 64 --train_epochs 15
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (seed_everything, get_device, prRed, prGreen, load_cifar10,
                   ResNet18_client_side, ResNet18_server_side)

Q2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(Q2_DIR, "results")
DEFAULT_DATA_DIR = os.path.join(Q2_DIR, "data")  # absolute => runnable from any cwd

# CIFAR-10: vehicles vs animals — a natural binary task for the norm-leak AUC.
VEHICLES = {0, 1, 8, 9}   # airplane, automobile, ship, truck
# (everything else — bird, cat, deer, dog, frog, horse — is an animal)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_device()
    return torch.device(name)


def train_split_model(client, server, train_loader, device, epochs, lr):
    """co-train client+server through the cut, standard split learning."""
    client.train()
    server.train()
    opt_c = torch.optim.Adam(client.parameters(), lr=lr)
    opt_s = torch.optim.Adam(server.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_c.zero_grad()
            opt_s.zero_grad()
            logits = server(client(x))
            loss = criterion(logits, y)
            loss.backward()
            opt_c.step()
            opt_s.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        prRed(f"  split-train epoch {epoch + 1}/{epochs}  loss {running / n:.4f}")


@torch.no_grad()
def _accuracy(server, client, loader, device) -> float:
    client.eval()
    server.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = server(client(x)).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def capture_cut_gradients(client, server, loader, device, limit):
    """returns (G, Y): flattened per-sample gradients at the cut + true labels.

    eval() matters here: batchnorm has to use running stats or each sample's
    gradient depends on its batchmates and they aren't per-sample anymore.
    """
    client.eval()
    server.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    grads, labels = [], []
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = client(x).detach().requires_grad_(True)
        loss = criterion(server(z), y)   # sum reduction => z.grad[i] = d loss_i / d z[i]
        loss.backward()
        grads.append(z.grad.detach().flatten(1).cpu())
        labels.append(y.cpu())
        seen += x.size(0)
        if seen >= limit:
            break
    G = torch.cat(grads)[:limit]
    Y = torch.cat(labels)[:limit]
    return G, Y


def norm_leak_auc(G: torch.Tensor, Y: torch.Tensor) -> float:
    """Li et al. norm attack: AUC of ||cut gradient|| vs the binary label.

    reported in [0.5, 1.0] -- the attacker can always flip its decision, so
    AUC below 0.5 just counts as 1-AUC.
    """
    scores = G.norm(dim=1).numpy()
    binary = np.array([0 if int(c) in VEHICLES else 1 for c in Y.numpy()])
    pos, neg = scores[binary == 1], scores[binary == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    auc = (ranks[binary == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(max(auc, 1.0 - auc))


def probe_label_accuracy(G, Y, device, num_classes=10, test_frac=0.3,
                         epochs=300, lr=0.01, wd=1e-4) -> float:
    """supervised probe: logistic regression, cut gradient -> class. top-1 acc."""
    n = G.size(0)
    n_test = int(n * test_frac)
    perm = torch.randperm(n)
    te, tr = perm[:n_test], perm[n_test:]
    Xtr, Ytr = G[tr].to(device), Y[tr].to(device)
    Xte, Yte = G[te].to(device), Y[te].to(device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    clf = nn.Linear(Xtr.size(1), num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(clf(Xtr), Ytr).backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(Xte).argmax(1) == Yte).float().mean().item()
    return 100.0 * acc


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
    p.add_argument("--train_epochs", type=int, default=15,
                   help="Epochs to train the split model before capturing gradients")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--attack_samples", type=int, default=4000,
                   help="How many test samples to capture cut gradients for")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | mps | cuda")
    p.add_argument("--method", type=str, default="splitnn",
                   help="Label for the output filename")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR)
    args = p.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"=== Label leakage | cut={args.cut_layer} seed={args.seed} "
          f"bc={args.base_channels} device={device} ===")

    train_set, test_set = load_cifar10(args.data_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # Attack data: clean (test-transform) images, no shuffle needed.
    attack_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    client = ResNet18_client_side(cut_layer=args.cut_layer,
                                  base_channels=args.base_channels).to(device)
    server = ResNet18_server_side(cut_layer=args.cut_layer,
                                  base_channels=args.base_channels).to(device)

    train_split_model(client, server, train_loader, device,
                      args.train_epochs, args.lr)
    acc = _accuracy(server, client, attack_loader, device)
    prGreen(f"  split model test accuracy: {acc:.2f}%")

    G, Y = capture_cut_gradients(client, server, attack_loader, device,
                                 args.attack_samples)
    print(f"  captured {G.size(0)} cut gradients of dim {G.size(1)}")

    auc = norm_leak_auc(G, Y)
    probe_acc = probe_label_accuracy(G, Y, device)
    chance = 100.0 / 10

    prGreen(f"--- cut={args.cut_layer}: norm-leak AUC {auc:.4f} "
            f"(0.5=no leak) | probe label acc {probe_acc:.2f}% (chance {chance:.0f}%)")

    out_path = os.path.join(
        args.output_dir,
        f"labelleak_{args.method}_cut{args.cut_layer}_seed{args.seed}.csv")
    save_summary_csv(out_path, {
        "method": args.method,
        "cut_layer": args.cut_layer,
        "seed": args.seed,
        "base_channels": args.base_channels,
        "train_epochs": args.train_epochs,
        "model_test_acc": round(acc, 4),
        "norm_leak_auc": round(auc, 4),
        "probe_label_acc": round(probe_acc, 4),
        "chance_acc": chance,
    })
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
