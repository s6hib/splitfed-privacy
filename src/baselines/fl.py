# ===========================================================
# Federated Learning: ResNet18 on CIFAR-10
# Adapted from: SplitFed — Thapa et al., AAAI 2022
# ===========================================================
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (seed_everything, get_device, prRed, prGreen, load_cifar10,
                   ResNet18, BasicBlock, calculate_accuracy,
                   DatasetSplit, dataset_iid, FedAvg, save_results_csv)

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)

# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--num_users', type=int, default=5)
parser.add_argument('--frac', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--base_channels', type=int, default=16,
                    help='Base channel width for ResNet18 (16=lightweight/CPU, 64=standard)')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--output_dir', type=str, default=DEFAULT_RESULTS_DIR)
args = parser.parse_args()

# ============================================================================
seed_everything(args.seed)
program = "FL ResNet18 on CIFAR10"
print(f"---------{program}----------")

device = get_device()

# ============================================================================
#  Client
# ============================================================================
class LocalUpdate(object):
    def __init__(self, idx, lr, device, dataset_train, dataset_test, idxs, idxs_test):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs),
                                     batch_size=args.batch_size, shuffle=True)
        self.ldr_test  = DataLoader(DatasetSplit(dataset_test, idxs_test),
                                     batch_size=args.batch_size, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        epoch_acc, epoch_loss = [], []
        for _ in range(self.local_ep):
            batch_acc, batch_loss = [], []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                fx = net(images)
                loss = self.loss_func(fx, labels)
                acc = calculate_accuracy(fx, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            prRed('Client{} Train => Local Epoch: {}  Acc: {:.3f}  Loss: {:.4f}'.format(
                self.idx, _, epoch_acc[-1], epoch_loss[-1]))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

    def evaluate(self, net):
        net.eval()
        epoch_acc, epoch_loss = [], []
        with torch.no_grad():
            batch_acc, batch_loss = [], []
            for images, labels in self.ldr_test:
                images, labels = images.to(self.device), labels.to(self.device)
                fx = net(images)
                loss = self.loss_func(fx, labels)
                acc = calculate_accuracy(fx, labels)
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            prGreen('Client{} Test =>                  Loss: {:.4f}  Acc: {:.3f}'.format(
                self.idx, epoch_loss[-1], epoch_acc[-1]))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)


# ============================================================================
#  Data
# ============================================================================
dataset_train, dataset_test = load_cifar10()
dict_users = dataset_iid(dataset_train, args.num_users)
dict_users_test = dataset_iid(dataset_test, args.num_users)

# ============================================================================
#  Model
# ============================================================================
net_glob = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=10,
                    base_channels=args.base_channels)
net_glob.to(device)
net_glob.train()
w_glob = net_glob.state_dict()

loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []

# ============================================================================
#  Training loop
# ============================================================================
for iter in range(args.epochs):
    w_locals, loss_locals_train, acc_locals_train = [], [], []
    loss_locals_test, acc_locals_test = [], []
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    for idx in idxs_users:
        local = LocalUpdate(idx, args.lr, device,
                            dataset_train=dataset_train, dataset_test=dataset_test,
                            idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        w, loss_train, acc_train = local.train(net=copy.deepcopy(net_glob).to(device))
        w_locals.append(copy.deepcopy(w))
        loss_locals_train.append(loss_train)
        acc_locals_train.append(acc_train)

        loss_test, acc_test = local.evaluate(net=copy.deepcopy(net_glob).to(device))
        loss_locals_test.append(loss_test)
        acc_locals_test.append(acc_test)

    # Federation
    w_glob = FedAvg(w_locals)
    print("------------------------------------------------")
    print("------ Federation process at Server-Side -------")
    print("------------------------------------------------")
    net_glob.load_state_dict(w_glob)

    acc_avg_train = sum(acc_locals_train) / len(acc_locals_train)
    acc_train_collect.append(acc_avg_train)
    acc_avg_test = sum(acc_locals_test) / len(acc_locals_test)
    acc_test_collect.append(acc_avg_test)
    loss_avg_train = sum(loss_locals_train) / len(loss_locals_train)
    loss_train_collect.append(loss_avg_train)
    loss_avg_test = sum(loss_locals_test) / len(loss_locals_test)
    loss_test_collect.append(loss_avg_test)

    print('--- SERVER ---')
    print('Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(
        iter, acc_avg_train, loss_avg_train))
    print('Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(
        iter, acc_avg_test, loss_avg_test))
    print('---')

print("Training and Evaluation completed!")

rounds = list(range(1, len(acc_train_collect) + 1))
os.makedirs(args.output_dir, exist_ok=True)
out_path = os.path.join(args.output_dir, f"{program}_seed{args.seed}.csv")
save_results_csv(out_path, rounds, acc_train_collect, acc_test_collect,
                 loss_train_collect, loss_test_collect)
print(f"Results saved to {out_path}")
