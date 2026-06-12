# ===========================================================
# Centralized (normal) learning: ResNet18 on CIFAR-10
# Adapted from: SplitFed — Thapa et al., AAAI 2022
# ===========================================================
import argparse
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (seed_everything, get_device, prRed, prGreen, load_cifar10,
                   ResNet18, BasicBlock, calculate_accuracy, save_results_csv)

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)

# ============================================================================
#  Args
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--base_channels', type=int, default=16,
                    help='Base channel width for ResNet18 (16=lightweight/CPU, 64=standard)')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--output_dir', type=str, default=DEFAULT_RESULTS_DIR)
args = parser.parse_args()

# ============================================================================
seed_everything(args.seed)
program = "Normal Learning ResNet18 on CIFAR10"
print(f"---------{program}----------")

device = get_device()

# ============================================================================
#  Data
# ============================================================================
dataset_train, dataset_test = load_cifar10()
train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
test_loader  = DataLoader(dataset_test,  batch_size=args.batch_size, shuffle=False)

print(f'Training samples: {len(dataset_train)}, Test samples: {len(dataset_test)}')

# ============================================================================
#  Model
# ============================================================================
net_glob = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=10,
                    base_channels=args.base_channels)
net_glob.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)

# ============================================================================
#  Training & Evaluation
# ============================================================================
loss_train_collect = []
loss_test_collect = []
acc_train_collect = []
acc_test_collect = []

def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (x, y) in iterator:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        fx = model(x)
        loss = criterion(fx, y)
        acc = calculate_accuracy(fx, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x, y = x.to(device), y.to(device)
            fx = model(x)
            loss = criterion(fx, y)
            acc = calculate_accuracy(fx, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


start_time = time.time()
for epoch in range(args.epochs):
    train_loss, train_acc = train(net_glob, device, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(net_glob, device, test_loader, criterion)

    loss_train_collect.append(train_loss)
    loss_test_collect.append(test_loss)
    acc_train_collect.append(train_acc)
    acc_test_collect.append(test_acc)

    prRed(f'Train => Epoch: {epoch}  Acc: {train_acc:05.2f}%  Loss: {train_loss:.3f}')
    prGreen(f'Test  =>              Acc: {test_acc:05.2f}%  Loss: {test_loss:.3f}')

elapsed = (time.time() - start_time) / 60
print(f'\nTotal Training Time: {elapsed:.2f} min')
print("Training and Evaluation completed!")

# ============================================================================
#  Save results
# ============================================================================
rounds = list(range(1, len(acc_train_collect) + 1))
os.makedirs(args.output_dir, exist_ok=True)
out_path = os.path.join(args.output_dir, f"{program}_seed{args.seed}.csv")
save_results_csv(out_path, rounds, acc_train_collect, acc_test_collect,
                 loss_train_collect, loss_test_collect)
print(f"Results saved to {out_path}")
