"""A small U-Net that corrects frozen DAV2 predictions in log-depth space."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
    )


class DepthCorrectionNet(nn.Module):
    requires_pseudo_depth = True

    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.DEPTH.CORRECTION.CHANNELS
        if channels < 8 or channels % 8:
            raise ValueError('Correction channels must be a positive multiple of 8.')
        self.scales = cfg.DATASET.SCALES
        self.min_depth = cfg.MODEL.DEPTH.MIN_DEPTH
        self.max_depth = cfg.MODEL.DEPTH.MAX_DEPTH
        self.max_log_residual = cfg.MODEL.DEPTH.CORRECTION.MAX_LOG_RESIDUAL
        if not 0 < self.min_depth < self.max_depth or self.max_log_residual <= 0:
            raise ValueError('Invalid correction depth range or residual bound.')
        self.enc0 = conv_block(5, channels)
        self.enc1 = conv_block(channels, channels * 2)
        self.enc2 = conv_block(channels * 2, channels * 4)
        self.bridge = conv_block(channels * 4, channels * 8)
        self.dec2 = conv_block(channels * 12, channels * 4)
        self.dec1 = conv_block(channels * 6, channels * 2)
        self.dec0 = conv_block(channels * 3, channels)
        self.residual = nn.Conv2d(channels, 1, 3, padding=1)
        # Start from the decoded metric teacher, not random depth.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def up_cat(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                                        align_corners=False), skip], dim=1)

    def forward(self, img, daytime=None, metric=None, relative=None):
        if metric is None or relative is None:
            raise ValueError('DAV2 correction requires both metric and relative pseudo-depth maps.')
        if metric.shape != relative.shape or metric.shape != img[:, :1].shape:
            raise ValueError('RGB and both pseudo-depth maps must have matching batch/spatial sizes.')
        metric = metric.detach().clamp(self.min_depth, self.max_depth)
        relative = relative.detach()
        x = torch.cat([(img - 0.45) / 0.225,
                       metric.log() / math.log(self.max_depth / self.min_depth),
                       relative * 2.0 - 1.0], dim=1)
        x0 = self.enc0(x)
        x1 = self.enc1(F.avg_pool2d(x0, 2))
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.bridge(F.avg_pool2d(x2, 2))
        x = self.dec2(self.up_cat(x3, x2))
        x = self.dec1(self.up_cat(x, x1))
        x = self.dec0(self.up_cat(x, x0))
        bound = self.max_log_residual
        residual = bound * torch.tanh(self.residual(x) / bound)
        depth = (metric * residual.exp()).clamp(self.min_depth, self.max_depth)
        inv_depth = depth.reciprocal()
        height, width = img.shape[-2:]
        return {('disp', 0, scale): inv_depth if scale == 0 else F.interpolate(
            inv_depth, size=(height // 2 ** scale, width // 2 ** scale),
            mode='bilinear', align_corners=False) for scale in self.scales}
