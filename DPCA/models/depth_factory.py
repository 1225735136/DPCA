"""Shared depth forwarding keeps pseudo labels aligned during flip augmentation."""

from models.depth_net import DepthNet
from models.depth_correction import DepthCorrectionNet


def build_depth_model(cfg, model_type=None, initialize_pretrained=True):
    model_type = model_type or cfg.MODEL.DEPTH.TYPE
    if model_type == 'baseline':
        return DepthNet(cfg)
    if model_type == 'dav2_correction':
        return DepthCorrectionNet(cfg)
    if model_type == 'dav2_small':
        from models.dav2_student import DAV2Student
        return DAV2Student(cfg, initialize_pretrained=initialize_pretrained)
    raise ValueError('Unknown depth model: {}'.format(model_type))


def forward_depth(model, batch, color_key='color', weather_key='weather', flip=False):
    def transform(value):
        return value.flip(-1) if flip else value

    image = transform(batch[(color_key, 0)])
    kwargs = {}
    if getattr(model, 'requires_pseudo_depth', False):
        for kind in ('metric', 'relative'):
            key = ('pseudo_' + kind, 0)
            if key not in batch:
                raise ValueError('Missing {}. Enable DATASET.PSEUDO_DEPTH and set both paths.'.format(key))
            kwargs[kind] = transform(batch[key])
    return model(image, batch.get(weather_key), **kwargs)
