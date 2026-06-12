# utils.py — shared models, data loading, and helpers for every method script.
# the model/data code is carried over from my Q1 port of the original SplitFed
# repo (Thapa et al. AAAI 2022); see the README for what changed vs upstream.

import os

# let unsupported ops fall back to CPU on Apple Silicon (MPS) instead of
# erroring out. has to be set before torch is imported. harmless on CUDA/CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import math
import csv

# ============================================================================
#  Reproducibility
# ============================================================================
SEED = 1234

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


# ============================================================================
#  Device selection
# ============================================================================
def get_device(verbose=True):
    """CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
    return device


# ============================================================================
#  Colored console output (matches original repo style)
# ============================================================================
def prRed(skk):    print("\033[91m {}\033[00m".format(skk))
def prGreen(skk):  print("\033[92m {}\033[00m".format(skk))


# ============================================================================
#  CIFAR-10 Data Loading
# ============================================================================
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


def load_cifar10(data_dir="./data"):
    dataset_train = datasets.CIFAR10(root=data_dir, train=True,
                                      download=True, transform=train_transforms)
    dataset_test  = datasets.CIFAR10(root=data_dir, train=False,
                                      download=True, transform=test_transforms)
    return dataset_train, dataset_test


# ============================================================================
#  IID Data Partitioning (matches original repo)
# ============================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# ============================================================================
#  Accuracy Calculation
# ============================================================================
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# ============================================================================
#  Federated Averaging
# ============================================================================
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# ============================================================================
#  ResNet18 — Full model (for Centralized and FL)
# ============================================================================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=10, base_channels=64):
        # base_channels controls model width: 64=standard ResNet18
        c = base_channels
        self.inplanes = c
        super(ResNet18, self).__init__()
        # CIFAR-adapted: 3x3 conv stride 1, no maxpool (standard for 32x32)
        self.conv1 = nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, c, layers[0])
        self.layer2 = self._make_layer(block, c*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, c*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, c*8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
#  Split ResNet18 — Client & Server models with configurable cut layer
#
#  cut_layer=1 : client gets [conv1+bn+relu],              server gets [layer1..layer4+fc]
#  cut_layer=2 : client gets [conv1+bn+relu+layer1],       server gets [layer2..layer4+fc]  (original repo default)
#  cut_layer=3 : client gets [conv1+bn+relu+layer1+2],     server gets [layer3..layer4+fc]
#  cut_layer=4 : client gets [conv1+bn+relu+layer1+2+3],   server gets [layer4+fc]
# ============================================================================

class ResNet18_client_side(nn.Module):
    def __init__(self, cut_layer=2, num_classes=10, base_channels=64):
        super(ResNet18_client_side, self).__init__()
        self.cut_layer = cut_layer
        c = base_channels
        self.inplanes = c

        # All clients have conv1+bn+relu
        self.conv1 = nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

        if cut_layer >= 2:
            self.layer1 = self._make_layer(BasicBlock, c, 2)
        if cut_layer >= 3:
            self.layer2 = self._make_layer(BasicBlock, c*2, 2, stride=2)
        if cut_layer >= 4:
            self.layer3 = self._make_layer(BasicBlock, c*4, 2, stride=2)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.cut_layer >= 2:
            x = self.layer1(x)
        if self.cut_layer >= 3:
            x = self.layer2(x)
        if self.cut_layer >= 4:
            x = self.layer3(x)
        return x


class ResNet18_server_side(nn.Module):
    def __init__(self, cut_layer=2, num_classes=10, base_channels=64):
        super(ResNet18_server_side, self).__init__()
        self.cut_layer = cut_layer
        c = base_channels
        # Determine input planes based on what the client outputs
        if cut_layer == 1:
            self.inplanes = c
        elif cut_layer == 2:
            self.inplanes = c
        elif cut_layer == 3:
            self.inplanes = c*2
        elif cut_layer == 4:
            self.inplanes = c*4

        if cut_layer <= 1:
            self.layer1 = self._make_layer(BasicBlock, c, 2)
        if cut_layer <= 2:
            self.layer2 = self._make_layer(BasicBlock, c*2, 2, stride=2)
        if cut_layer <= 3:
            self.layer3 = self._make_layer(BasicBlock, c*4, 2, stride=2)
        # layer4 is always on the server
        self.layer4 = self._make_layer(BasicBlock, c*8, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c*8 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.cut_layer <= 1:
            x = self.layer1(x)
        if self.cut_layer <= 2:
            x = self.layer2(x)
        if self.cut_layer <= 3:
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
#  Communication cost helpers
# ============================================================================
def get_activation_size(cut_layer, batch_size=256, base_channels=64):
    """Return activation tensor size in number of floats for one batch."""
    # Feature map spatial sizes for CIFAR-10 (32x32 input, stride-1 conv1)
    c = base_channels
    sizes = {
        1: (32, 32, c),       # After conv1+bn+relu
        2: (32, 32, c),       # After layer1
        3: (16, 16, c*2),     # After layer2
        4: (8, 8, c*4),       # After layer3
    }
    h, w, ch = sizes[cut_layer]
    return batch_size * h * w * ch


def get_activation_mb(cut_layer, batch_size=256, base_channels=64):
    """Return activation size in MB (float32) for one batch."""
    return get_activation_size(cut_layer, batch_size, base_channels) * 4 / (1024 * 1024)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================================
#  Model checkpointing (used by the Q2 feature-inversion attack)
# ============================================================================
def save_client_checkpoint(net_client, method, cut_layer, base_channels, seed, ckpt_dir):
    """save a trained client encoder so the inversion attack can load it later.

    stores cut_layer/base_channels next to the weights so the attack can
    rebuild the right architecture. weights go to CPU so it loads anywhere.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{method}_client_cut{cut_layer}_seed{seed}.pt")
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in net_client.state_dict().items()},
        "method": method,
        "cut_layer": cut_layer,
        "base_channels": base_channels,
        "seed": seed,
    }, path)
    return path


# ============================================================================
#  Result saving
# ============================================================================
def save_results_csv(filename, round_list, acc_train, acc_test,
                     loss_train=None, loss_test=None):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['round', 'acc_train', 'acc_test']
        if loss_train is not None:
            header += ['loss_train', 'loss_test']
        writer.writerow(header)
        for i in range(len(round_list)):
            row = [round_list[i], acc_train[i], acc_test[i]]
            if loss_train is not None:
                row += [loss_train[i], loss_test[i]]
            writer.writerow(row)
