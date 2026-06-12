# decoder for the inversion attack (step 2). takes the smashed data the client
# sends across the cut and tries to rebuild the original 32x32 image. needs a
# separate decoder per cut layer since the activation shape changes:
#   cut 1/2: (c, 32, 32)   cut 3: (2c, 16, 16)   cut 4: (4c, 8, 8)
# output goes through a sigmoid into [0,1] so PSNR/SSIM can compare it straight
# against the un-normalized image.
import math

import torch
from torch import nn


def smashed_shape(cut_layer, base_channels):
    """(channels, h, w) of the client output at each cut. matches utils.get_activation_size."""
    c = base_channels
    shapes = {
        1: (c,     32, 32),
        2: (c,     32, 32),
        3: (c * 2, 16, 16),
        4: (c * 4, 8,  8),
    }
    if cut_layer not in shapes:
        raise ValueError(f"cut_layer must be in {{1,2,3,4}}, got {cut_layer}")
    return shapes[cut_layer]


class InversionDecoder(nn.Module):
    """smashed data -> image in [0,1].

    upsampling is nearest-neighbour + conv instead of transposed convs, which
    kept giving checkerboard artifacts. number of upsample stages comes from
    the input size so one class handles all four cuts.
    """

    def __init__(self, in_channels, in_hw, out_hw=32, out_channels=3, width=128):
        super().__init__()
        if in_hw > out_hw or out_hw % in_hw != 0:
            raise ValueError(
                f"out_hw ({out_hw}) must be a power-of-two multiple of in_hw ({in_hw})"
            )
        n_up = int(round(math.log2(out_hw / in_hw)))

        layers = [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        ]

        ch = width
        for _ in range(n_up):
            out_ch = max(ch // 2, 32)
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch

        # refine at full res, then the image head
        layers += [
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_decoder(cut_layer, base_channels, width=128):
    """decoder sized for the given cut layer."""
    in_channels, in_hw, _ = smashed_shape(cut_layer, base_channels)
    return InversionDecoder(in_channels=in_channels, in_hw=in_hw, width=width)
