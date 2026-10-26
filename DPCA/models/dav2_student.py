"""RGB-only DAV2 Small metric student with the md4all inverse-depth interface."""

import importlib
from pathlib import Path
import sys
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F


def metric_model_class():
    # Isolate the metric package from the identically named relative-depth package.
    package_name = '_md4all_dav2_metric'
    package_path = Path(__file__).resolve().parents[1] / 'generate-pseudo/metric_depth/depth_anything_v2'
    if package_name not in sys.modules:
        if not (package_path / 'dpt.py').is_file():
            raise ImportError('DAV2 metric source package is missing: {}'.format(package_path))
        package = ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(package_name + '.dpt').DepthAnythingV2


class DAV2Student(nn.Module):
    requires_pseudo_depth = False

    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.scales = list(cfg.DATASET.SCALES)
        self.min_depth = cfg.MODEL.DEPTH.MIN_DEPTH
        self.max_depth = cfg.MODEL.DEPTH.MAX_DEPTH
        if not 0 < self.min_depth < self.max_depth:
            raise ValueError('Invalid DAV2 student depth range.')
        self.network = metric_model_class()(
            encoder='vits', features=64, out_channels=[48, 96, 192, 384],
            max_depth=self.max_depth)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if initialize_pretrained:
            path = cfg.MODEL.DEPTH.DAV2.PRETRAINED_PATH
            if not path or not Path(path).is_file():
                raise FileNotFoundError('DAV2 Small metric weights not found: {}. Set MODEL.DEPTH.DAV2.PRETRAINED_PATH.'.format(path))
            state = torch.load(path, map_location='cpu', weights_only=True)
            # Strict loading prevents accidentally initializing from a different size.
            self.network.load_state_dict(state, strict=True)

    def optimizer_param_groups(self, lr, encoder_lr_factor):
        if not 0 < encoder_lr_factor <= 1:
            raise ValueError('DAV2 encoder LR factor must be in (0, 1].')
        return [
            {'params': self.network.pretrained.parameters(), 'lr': lr * encoder_lr_factor},
            {'params': self.network.depth_head.parameters(), 'lr': lr},
        ]

    def forward(self, img, daytime=None):
        height, width = img.shape[-2:]
        # Replicate padding preserves RGB geometry and avoids cropping border pixels.
        x = (img - self.image_mean) / self.image_std
        x = F.pad(x, (0, (-width) % 14, 0, (-height) % 14), mode='replicate')
        depth = self.network(x)[:, None, :height, :width]
        inv_depth = depth.clamp(self.min_depth, self.max_depth).reciprocal()
        return {
            ('disp', 0, scale): inv_depth if scale == 0 else F.interpolate(
                inv_depth, size=(height // 2 ** scale, width // 2 ** scale),
                mode='bilinear', align_corners=False)
            for scale in self.scales
        }
