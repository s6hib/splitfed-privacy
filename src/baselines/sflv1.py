# ============================================================================
# SplitFedV1 (SFLV1) Learning: ResNet18 on CIFAR-10
# Adapted from: SplitFed — Thapa et al., AAAI 2022
#
# SFLV1 = Split Learning + FedAvg on BOTH client-side and server-side models
# Each client gets its own server-side model copy; after all clients train,
# both client and server models are averaged.
# ============================================================================
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
                   ResNet18_client_side, ResNet18_server_side,
                   calculate_accuracy, DatasetSplit, dataset_iid,
                   FedAvg, save_results_csv, save_client_checkpoint)

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
parser.add_argument('--cut_layer', type=int, default=2, choices=[1,2,3,4])
parser.add_argument('--base_channels', type=int, default=16,
                    help='Base channel width for ResNet18 (16=lightweight/CPU, 64=standard)')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--output_dir', type=str, default=DEFAULT_RESULTS_DIR)
parser.add_argument('--save_client_ckpt', action='store_true',
                    help='Save the trained client-side encoder for the Q2 inversion attack')
parser.add_argument('--ckpt_dir', type=str,
                    default=os.path.join(DEFAULT_RESULTS_DIR, 'checkpoints'))
args = parser.parse_args()

# ============================================================================
seed_everything(args.seed)
program = "SFLV1 ResNet18 on CIFAR10"
print(f"---------{program}----------")

device = get_device()

# ============================================================================
#  Models
# ============================================================================
net_glob_client = ResNet18_client_side(cut_layer=args.cut_layer,
                                       base_channels=args.base_channels)
net_glob_client.to(device)

net_glob_server = ResNet18_server_side(cut_layer=args.cut_layer,
                                       base_channels=args.base_channels)
net_glob_server.to(device)

print(f"Client model params: {sum(p.numel() for p in net_glob_client.parameters()):,}")
print(f"Server model params: {sum(p.numel() for p in net_glob_server.parameters()):,}")

# ============================================================================
#  Server-side globals
# ============================================================================
criterion = nn.CrossEntropyLoss()
lr = args.lr
num_users = args.num_users

loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []
count1 = 0
count2 = 0

acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
w_locals_server = []

idx_collect = []
l_epoch_check = False
fed_check = False

# SFLV1: each client has its own server-side model copy
net_model_server = [net_glob_server for _ in range(num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(device)


def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, device, batch_acc_train, batch_loss_train
    global l_epoch_check, fed_check, count1, idx_collect
    global loss_train_collect, acc_train_collect
    global acc_avg_all_user_train, loss_avg_all_user_train
    global loss_train_collect_user, acc_train_collect_user
    global w_locals_server, w_glob_server, net_server, net_glob_server

    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr)

    optimizer_server.zero_grad()
    fx_client = fx_client.to(device)
    y = y.to(device)

    fx_server = net_server(fx_client)
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # Update per-client server model
    net_model_server[idx] = copy.deepcopy(net_server)

    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)
        batch_acc_train.clear()
        batch_loss_train.clear()
        count1 = 0

        prRed('Client{} Train => Local Epoch: {}  Acc: {:.3f}  Loss: {:.4f}'.format(
            idx, l_epoch_count, acc_avg_train, loss_avg_train))

        w_server = net_server.state_dict()

        if l_epoch_count == l_epoch - 1:
            l_epoch_check = True
            w_locals_server.append(copy.deepcopy(w_server))
            loss_train_collect_user.append(loss_avg_train)
            acc_train_collect_user.append(acc_avg_train)
            if idx not in idx_collect:
                idx_collect.append(idx)

        if len(idx_collect) == num_users:
            fed_check = True
            # Server-side federation
            w_glob_server = FedAvg(w_locals_server)
            net_glob_server.load_state_dict(w_glob_server)
            net_model_server = [net_glob_server for _ in range(num_users)]
            w_locals_server = []
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            acc_train_collect_user.clear()
            loss_train_collect_user.clear()

    return dfx_client


def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2
    global l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user
    global acc_avg_all_user_train, loss_avg_all_user_train

    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        fx_server = net(fx_client)
        loss = criterion(fx_server, y)
        acc = calculate_accuracy(fx_server, y)
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)
            batch_acc_test.clear()
            batch_loss_test.clear()
            count2 = 0

            prGreen('Client{} Test =>                  Acc: {:.3f}  Loss: {:.4f}'.format(
                idx, acc_avg_test, loss_avg_test))

            if l_epoch_check:
                l_epoch_check = False
                loss_test_collect_user.append(loss_avg_test)
                acc_test_collect_user.append(acc_avg_test)

            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")
                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user.clear()
                loss_test_collect_user.clear()

                print("====================== SFLV1 SERVER ======================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(
                    ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(
                    ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")


# ============================================================================
#  Client
# ============================================================================
class Client(object):
    def __init__(self, net_client_model, idx, lr, device,
                 dataset_train, dataset_test, idxs, idxs_test):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs),
                                     batch_size=args.batch_size, shuffle=True)
        self.ldr_test  = DataLoader(DatasetSplit(dataset_test, idxs_test),
                                     batch_size=args.batch_size, shuffle=True)

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
        for ep in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for images, labels in self.ldr_train:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                dfx = train_server(client_fx, labels, ep, self.local_ep, self.idx, len_batch)
                fx.backward(dfx)
                optimizer_client.step()
        return net.state_dict()

    def evaluate(self, net, ell):
        net.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for images, labels in self.ldr_test:
                images, labels = images.to(self.device), labels.to(self.device)
                fx = net(images)
                evaluate_server(fx, labels, self.idx, len_batch, ell)


# ============================================================================
#  Data
# ============================================================================
dataset_train, dataset_test = load_cifar10()
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

# ============================================================================
#  Training loop — SFLV1: FedAvg on both client & server
# ============================================================================
net_glob_client.train()
w_glob_client = net_glob_client.state_dict()

for iter in range(args.epochs):
    m = max(int(args.frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []

    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device,
                       dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

    # Client-side federation
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    w_glob_client = FedAvg(w_locals_client)
    net_glob_client.load_state_dict(w_glob_client)

print("Training and Evaluation completed!")
rounds = list(range(1, len(acc_train_collect) + 1))
os.makedirs(args.output_dir, exist_ok=True)
out_path = os.path.join(args.output_dir, f"{program}_cut{args.cut_layer}_seed{args.seed}.csv")
save_results_csv(out_path, rounds, acc_train_collect, acc_test_collect,
                 loss_train_collect, loss_test_collect)
print(f"Results saved to {out_path}")

if args.save_client_ckpt:
    ckpt_path = save_client_checkpoint(net_glob_client, "sflv1", args.cut_layer,
                                       args.base_channels, args.seed, args.ckpt_dir)
    print(f"Client encoder checkpoint saved to {ckpt_path}")
