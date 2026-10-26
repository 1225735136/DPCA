"""Read DAV2 uint16 exports using their manifest, preserving metric units."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class PseudoDepthReader:
    def __init__(self, cfg):
        settings = cfg.DATASET.PSEUDO_DEPTH
        if not settings.METRIC_PATH or not settings.RELATIVE_PATH:
            raise ValueError('Both pseudo-depth paths are required.')
        self.roots = {'metric': Path(settings.METRIC_PATH), 'relative': Path(settings.RELATIVE_PATH)}
        expected = {
            'metric': ('metric_depth_uint16_png_fixed_scale_v1', 'metric_depth'),
            'relative': ('relative_inverse_depth_uint16_png_max_per_image_v1', 'relative_inverse_depth'),
        }
        for kind, root in self.roots.items():
            with (root / 'manifest.json').open() as stream:
                manifest = json.load(stream)
            if (manifest.get('format'), manifest.get('representation')) != expected[kind]:
                raise ValueError('Unsupported {} pseudo-depth encoding in {}'.format(kind, root))
            if manifest.get('geometry') != 'original_uncropped_image':
                raise ValueError('Pseudo depth must use original, uncropped image geometry.')
            if kind == 'metric':
                self.metric_scale = float(manifest['png_scale_m'])
                if not np.isfinite(self.metric_scale) or self.metric_scale <= 0:
                    raise ValueError('Invalid metric PNG scale.')
        crop = cfg.DATASET.AUGMENTATION.CROP
        self.crop = (crop.TOP, crop.LEFT, crop.HEIGHT, crop.WIDTH)
        resize = cfg.DATASET.AUGMENTATION.RESIZE
        self.size = (resize.HEIGHT, resize.WIDTH)

    def __call__(self, filename, original_size):
        result = {}
        top, left, height, width = self.crop
        for kind, root in self.roots.items():
            # Existing exports append .png to an RGB filename already ending in .png.
            path = root / filename
            if not path.is_file():
                path = Path(str(path) + '.png')
            if not path.is_file():
                raise FileNotFoundError('Missing {} pseudo depth: {}'.format(kind, path))
            with Image.open(path) as image:
                if image.mode not in ('I', 'I;16', 'I;16B', 'I;16L'):
                    raise ValueError('Expected uint16 depth, not an 8-bit preview: {}'.format(path))
                array = np.asarray(image)
            if array.ndim != 2 or array.dtype.kind not in 'ui' or array.max() > 65535:
                raise ValueError('Expected a single-channel uint16 PNG: {}'.format(path))
            if array.shape != original_size:
                raise ValueError('Pseudo/RGB geometry mismatch: {}: {} != {}'.format(
                    path, array.shape, original_size))
            if min(top, left) < 0 or top + height > array.shape[0] or left + width > array.shape[1]:
                raise ValueError('Crop extends beyond pseudo-depth image.')
            tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 65535.0
            if kind == 'metric':
                tensor *= self.metric_scale
                if (tensor <= 0).any():
                    raise ValueError('Metric pseudo depth contains zero/invalid pixels: {}'.format(path))
            tensor = tensor[:, :, top:top + height, left:left + width]
            if kind == 'relative':
                # Match the old SSI sky proxy: exact zeros before interpolation.
                result[('pseudo_sky_mask', 0)] = F.interpolate(
                    (tensor == 0).float(), size=self.size, mode='nearest')[0].bool()
            result[('pseudo_' + kind, 0)] = F.interpolate(
                tensor, size=self.size, mode='bilinear', align_corners=False)[0]
        return result
