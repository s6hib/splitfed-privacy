# reconstruction metrics for the inversion attack. images in [0,1], (N,3,H,W).
# higher PSNR/SSIM = better reconstruction = worse privacy.
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure


@torch.no_grad()
def psnr(pred, target, max_val=1.0, eps=1e-8):
    """mean PSNR in dB over the batch."""
    mse = F.mse_loss(pred, target, reduction="mean")
    return float(10.0 * torch.log10((max_val ** 2) / (mse + eps)))


@torch.no_grad()
def ssim(pred, target, data_range=1.0):
    """mean SSIM over the batch."""
    return float(structural_similarity_index_measure(pred, target, data_range=data_range))


@torch.no_grad()
def reconstruction_metrics(pred, target):
    """all three in a dict, handy for CSV rows."""
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "mse": float(F.mse_loss(pred, target, reduction="mean")),
    }
