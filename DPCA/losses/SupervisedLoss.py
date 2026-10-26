# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/losses/supervised_loss.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.depth_factory import build_depth_model, forward_depth


class LogDepthA1Loss(nn.Module):
    """Log-depth L1 plus a hinge penalty outside the delta1 threshold.

    Inputs are inverse depths; absolute log differences are identical for
    depth and inverse depth. This avoids unstable reciprocal conversions.
    """
    def __init__(self, threshold=1.25, weight=1.0, log_l1_weight=1.0):
        super().__init__()
        if not math.isfinite(threshold) or threshold <= 1:
            raise ValueError('A1 threshold must be finite and greater than 1.')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('A1 penalty weight must be finite and nonnegative.')
        if not math.isfinite(log_l1_weight) or log_l1_weight < 0:
            raise ValueError('Log-depth L1 weight must be finite and nonnegative.')
        self.log_threshold = math.log(threshold)
        self.weight = weight
        self.log_l1_weight = log_l1_weight

    def forward(self, pred, target):
        error = (pred.clamp_min(1e-6).log()
                 - target.detach().clamp_min(1e-6).log()).abs()
        return (self.log_l1_weight * error
                + self.weight * torch.relu(error - self.log_threshold)).mean()


class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.
        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.
        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted metric depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth metric depth map
        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        diff = (pred - gt).abs()
        huber_c = (self.threshold * diff.detach().max()).clamp_min(1e-6)
        loss = torch.where(diff <= huber_c, diff,
                           (diff.square() + huber_c.square()) / (2.0 * huber_c))
        return loss.mean()


class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss


def get_loss_func(supervised_method, a1_threshold=1.25, a1_weight=1.0,
                  log_l1_weight=1.0):
    """Determines the supervised loss to be used, given the supervised method."""
    if supervised_method == 'log_depth_a1':
        return LogDepthA1Loss(a1_threshold, a1_weight, log_l1_weight)
    elif supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))


class SupervisedLoss(nn.Module):
    """
    Supervised loss for inverse depth maps.
    Parameters
    """
    def __init__(self, cfg, is_train):
        super().__init__()
        self.supervised_method = cfg.LOSS.SUPERVISED.METHOD
        self.loss_func = get_loss_func(
            self.supervised_method, cfg.LOSS.SUPERVISED.A1_THRESHOLD,
            cfg.LOSS.SUPERVISED.A1_WEIGHT, cfg.LOSS.SUPERVISED.LOG_L1_WEIGHT)
        self.scales = cfg.DATASET.SCALES
        self.min_depth = cfg.MODEL.DEPTH.MIN_DEPTH
        self.max_depth = cfg.MODEL.DEPTH.MAX_DEPTH
        self.hard_region = cfg.LOSS.SUPERVISED.HARD_REGION
        self.optimization_step = 0
        self.ssi = cfg.LOSS.SUPERVISED.SSI
        if self.ssi.ENABLED:
            if not cfg.DATASET.PSEUDO_DEPTH.ENABLED or 0 not in self.scales:
                raise ValueError('SSI supervision requires pseudo depth and scale 0.')
            if not math.isfinite(self.ssi.WEIGHT) or self.ssi.WEIGHT < 0:
                raise ValueError('SSI weight must be finite and nonnegative.')
        self.boundary = cfg.LOSS.SUPERVISED.BOUNDARY
        if self.boundary.ENABLED:
            b = self.boundary
            if not cfg.DATASET.PSEUDO_DEPTH.ENABLED or 0 not in self.scales:
                raise ValueError('Boundary supervision requires pseudo depth and scale 0.')
            if not math.isfinite(b.FRACTION) or not 0 < b.FRACTION <= 1:
                raise ValueError('Boundary fraction must be in (0, 1].')
            if not math.isfinite(b.WEIGHT) or b.WEIGHT < 0:
                raise ValueError('Boundary weight must be finite and nonnegative.')
            if not math.isfinite(b.RELATIVE_MIN) or not 0 <= b.RELATIVE_MIN < 1:
                raise ValueError('Boundary relative cutoff must be in [0, 1).')
            for name in ('RADIUS', 'WARMUP_STEPS'):
                value = getattr(b, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError('Boundary {} must be a nonnegative integer.'.format(name))
        if self.hard_region.ENABLED:
            if not math.isfinite(self.hard_region.RELATIVE_MIN) or not 0 <= self.hard_region.RELATIVE_MIN < 1:
                raise ValueError('Relative inverse-depth cutoff must be in [0, 1).')
            if self.hard_region.EXCLUDE_RELATIVE_FAR and not cfg.DATASET.PSEUDO_DEPTH.ENABLED:
                raise ValueError('Relative far-region exclusion requires pseudo-depth loading.')
            if self.supervised_method != 'abs_rel' or 0 not in self.scales:
                raise ValueError('Hard-region loss requires abs_rel and output scale 0.')
            if not math.isfinite(self.hard_region.FRACTION) or not 0 < self.hard_region.FRACTION <= 1:
                raise ValueError('Hard-region fraction must be in (0, 1].')
            if not math.isfinite(self.hard_region.WEIGHT) or self.hard_region.WEIGHT < 0:
                raise ValueError('Hard-region weight must be finite and nonnegative.')
            if isinstance(self.hard_region.WARMUP_STEPS, bool) or not isinstance(self.hard_region.WARMUP_STEPS, int) or self.hard_region.WARMUP_STEPS < 0:
                raise ValueError('Hard-region warmup steps must be a nonnegative integer.')
        self.local_crop = cfg.LOSS.SUPERVISED.LOCAL_CROP
        self.region = cfg.LOSS.SUPERVISED.REGION
        if self.region.ENABLED:
            if self.supervised_method != 'abs_rel' or 0 not in self.scales:
                raise ValueError('Region supervision requires abs_rel and scale 0.')
            if not math.isfinite(self.region.WEIGHT) or self.region.WEIGHT < 0:
                raise ValueError('Region weight must be finite and nonnegative.')
            if not 0 < self.region.FRACTION <= 1:
                raise ValueError('Region fraction must be in (0, 1].')
            for name, minimum in (('GRID_SIZE', 1), ('RADIUS', 0)):
                value = getattr(self.region, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError('Invalid region {}.'.format(name))
        if self.local_crop.ENABLED:
            if self.local_crop.SAMPLING not in ('random', 'balanced_edges'):
                raise ValueError('Unknown local crop sampling mode.')
            if isinstance(self.local_crop.GRID_SIZE, bool) or not isinstance(self.local_crop.GRID_SIZE, int) or self.local_crop.GRID_SIZE < 1:
                raise ValueError('Local crop grid size must be a positive integer.')
            if not 0 < self.local_crop.EDGE_FRACTION <= 1:
                raise ValueError('Local crop edge fraction must be in (0, 1].')
            if cfg.MODEL.DEPTH.TYPE not in ('baseline', 'dav2_small') or self.supervised_method not in ('abs_rel', 'log_depth_a1'):
                raise ValueError('Local crop training requires an RGB baseline student and abs_rel or log_depth_a1.')
            if not 0 < self.local_crop.MIN_FRACTION <= self.local_crop.MAX_FRACTION <= 1:
                raise ValueError('Local crop fractions must satisfy 0 < min <= max <= 1.')
            if not 0 <= self.local_crop.PROBABILITY <= 1:
                raise ValueError('Local crop probability must be between 0 and 1.')
            if not math.isfinite(self.local_crop.WEIGHT) or self.local_crop.WEIGHT <= 0:
                raise ValueError('Local crop weight must be finite and positive.')
        self.metric_l1_weight = cfg.LOSS.SUPERVISED.METRIC_L1_WEIGHT
        if not math.isfinite(self.metric_l1_weight) or self.metric_l1_weight < 0:
            raise ValueError('Metric L1 weight must be finite and nonnegative.')
        # Teacher weights come from its checkpoint, never from student pretraining.
        teacher_cfg = cfg.clone()
        teacher_cfg.defrost()
        teacher_cfg.MODEL.DEPTH.ENCODER.PRETRAINED = False
        self.teacher_net = build_depth_model(
            teacher_cfg, cfg.LOSS.SUPERVISED.TEACHER_TYPE, initialize_pretrained=False)
        if is_train:
            try:
                checkpoint = torch.load(
                    cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATH,
                    map_location='cpu',
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATH,
                    map_location='cpu',
                )

            state_dict = checkpoint.get('state_dict', checkpoint)
            if any(key.startswith('depth_model.') for key in state_dict):
                state_dict = {key[len('depth_model.'):]: weight for key, weight in state_dict.items()
                              if key.startswith('depth_model.')}
            self.teacher_net.load_state_dict(state_dict)
        for param in self.teacher_net.parameters():
            param.requires_grad = False
        self.teacher_net.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher_net.eval()
        return self

    def calculate_loss(self, inv_depths, gt_inv_depths):
        """
        Calculate the supervised loss.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps
        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        # If using a sparse loss, mask invalid pixels for all scales
        if self.supervised_method.startswith('sparse'):
            for i in self.scales:
                mask = (gt_inv_depths[i] > 0.).detach()
                inv_depths[i] = inv_depths[i][mask]
                gt_inv_depths[i] = gt_inv_depths[i][mask]
        # Metric-depth losses operate in meters, while the networks emit inverse depth.
        if self.supervised_method == 'metric_l1' or self.supervised_method.endswith('berhu'):
            inv_min, inv_max = 1.0 / self.max_depth, 1.0 / self.min_depth
            pred_depths = [inv_depths[i].clamp(min=inv_min, max=inv_max).reciprocal()
                           for i in self.scales]
            target_depths = [gt_inv_depths[i].clamp(min=inv_min, max=inv_max).reciprocal()
                             for i in self.scales]
            return sum(self.loss_func(pred, target)
                       for pred, target in zip(pred_depths, target_depths)) / len(self.scales)

        if self.supervised_method == 'log_depth_a1' and self.metric_l1_weight > 0:
            inv_min, inv_max = 1.0 / self.max_depth, 1.0 / self.min_depth
            losses = []
            for i in self.scales:
                pred, target = inv_depths[i], gt_inv_depths[i].detach()
                pred_depth = pred.clamp(min=inv_min, max=inv_max).reciprocal()
                target_depth = target.clamp(min=inv_min, max=inv_max).reciprocal()
                metric_l1 = (pred_depth - target_depth).abs().mean()
                losses.append(self.loss_func(pred, target)
                              + self.metric_l1_weight * metric_l1)
            return sum(losses) / len(losses)

        # Return per-scale average loss
        return sum([self.loss_func(inv_depths[i], gt_inv_depths[i]) for i in self.scales]) / len(self.scales)

    @staticmethod
    def hard_region_abs_rel(pred, target, fraction, valid_mask=None):
        """Per-image top-k metric AbsRel, with inverse-depth inputs.

        Selection is detached; selected errors retain their gradients. The
        inverse prediction denominator is equivalent to teacher metric depth.
        """
        target = target.detach()
        if pred.shape != target.shape or pred.ndim != 4:
            raise ValueError('Hard-region inputs must share [B,C,H,W] dimensions.')
        error = ((pred - target).abs() / pred.clamp_min(1e-6)).flatten(1)
        if valid_mask is not None:
            if valid_mask.shape != pred.shape:
                raise ValueError('Hard-region mask must match prediction dimensions.')
            image_losses = []
            for values, mask in zip(error, valid_mask.detach().bool().flatten(1)):
                values = values[mask]
                if values.numel():
                    count = max(1, math.ceil(values.numel() * fraction))
                    indices = values.detach().topk(count, sorted=False).indices
                    image_losses.append(values[indices].mean())
            # All-sky images contribute no extra supervision or gradient.
            return torch.stack(image_losses).mean() if image_losses else error.sum() * 0.0
        count = max(1, math.ceil(error.shape[1] * fraction))
        indices = error.detach().topk(count, dim=1, sorted=False).indices
        return error.gather(1, indices).mean()

    @staticmethod
    def teacher_boundary_mask(target, valid_mask, fraction=0.1, radius=2):
        """Detached top teacher log-depth gradients; invalid pairs never seed edges."""
        target = target.detach()
        valid = valid_mask.detach().bool() & torch.isfinite(target) & (target > 0)
        log_depth = -target.clamp_min(1e-6).log()
        vx = valid[..., :, 1:] & valid[..., :, :-1]
        vy = valid[..., 1:, :] & valid[..., :-1, :]
        gx = F.pad(torch.where(vx, (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs(), 0.), (0,1,0,0))
        gy = F.pad(torch.where(vy, (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs(), 0.), (0,0,0,1))
        strength = gx + gy
        mask = torch.zeros_like(valid)
        for i in range(target.shape[0]):
            candidates = torch.nonzero((valid[i] & (strength[i] > 0)).flatten()).flatten()
            if candidates.numel():
                count = min(candidates.numel(), max(1, math.ceil(valid[i].sum().item() * fraction)))
                chosen = candidates[strength[i].flatten()[candidates].topk(count, sorted=False).indices]
                mask[i].view(-1)[chosen] = True
        if radius:
            mask = F.max_pool2d(mask.float(), 2 * radius + 1, 1, radius).bool()
        return mask & valid

    @staticmethod
    def boundary_gradient_loss(pred, target, valid_mask, fraction=0.1, radius=2):
        """Match signed log-depth differences across valid neighboring pixels."""
        target = target.detach()
        valid = valid_mask.bool() & torch.isfinite(target) & (target > 0)
        band = SupervisedLoss.teacher_boundary_mask(target, valid, fraction, radius)
        p, t = -pred.clamp_min(1e-6).log(), -target.clamp_min(1e-6).log()
        ex = ((p[..., :, 1:] - p[..., :, :-1]) - (t[..., :, 1:] - t[..., :, :-1])).abs()
        ey = ((p[..., 1:, :] - p[..., :-1, :]) - (t[..., 1:, :] - t[..., :-1, :])).abs()
        mx = (band[..., :, 1:] | band[..., :, :-1]) & valid[..., :, 1:] & valid[..., :, :-1]
        my = (band[..., 1:, :] | band[..., :-1, :]) & valid[..., 1:, :] & valid[..., :-1, :]
        per_image = []
        for i in range(pred.shape[0]):
            count = mx[i].sum() + my[i].sum()
            if count:
                per_image.append((ex[i][mx[i]].sum() + ey[i][my[i]].sum()) / count)
        return torch.stack(per_image).mean() if per_image else pred.sum() * 0.

    @staticmethod
    def region_abs_rel(pred, target, grid_size=4, fraction=0.1, radius=2):
        """Reweight teacher edge neighborhoods using existing scale-0 predictions."""
        target = target.detach()
        height, width = target.shape[-2:]
        rows, cols = min(grid_size, height), min(grid_size, width)
        image_losses = []
        for image in range(target.shape[0]):
            cells = []
            for row in range(rows):
                for col in range(cols):
                    ys = slice(row * height // rows, (row + 1) * height // rows)
                    xs = slice(col * width // cols, (col + 1) * width // cols)
                    t = target[image:image + 1, :, ys, xs]
                    p = pred[image:image + 1, :, ys, xs]
                    valid = torch.isfinite(t) & (t > 0)
                    mask = SupervisedLoss.teacher_boundary_mask(t, valid, fraction, radius)
                    if mask.any():
                        error = (p - t).abs() / p.clamp_min(1e-6)
                        cells.append(error[mask].mean())
            if cells:
                image_losses.append(torch.stack(cells).mean())
        return torch.stack(image_losses).mean() if image_losses else pred.sum() * 0.

    @staticmethod
    def crop_pair(images, teacher_inv_depth, min_fraction, max_fraction,
                  sampling='random', grid_size=4, edge_fraction=0.1):
        """Aligned random patches, resized without changing metric depth units."""
        if images.shape[0] != teacher_inv_depth.shape[0] or images.shape[-2:] != teacher_inv_depth.shape[-2:]:
            raise ValueError('Local RGB and teacher maps must share batch/spatial dimensions.')
        height, width = images.shape[-2:]
        image_patches, targets = [], []
        for image, target in zip(images, teacher_inv_depth.detach()):
            fraction = min_fraction + (max_fraction - min_fraction) * torch.rand(()).item()
            crop_h, crop_w = max(1, round(height * fraction)), max(1, round(width * fraction))
            top = torch.randint(height - crop_h + 1, ()).item()
            left = torch.randint(width - crop_w + 1, ()).item()
            if sampling == 'balanced_edges':
                # Choose a spatial cell uniformly before choosing a strong edge.
                # Large buildings cannot monopolize a global top-k selection.
                valid = torch.isfinite(target) & (target > 0)
                log_depth = -target.clamp_min(1e-6).log()
                gx = F.pad(torch.where(valid[..., :, 1:] & valid[..., :, :-1],
                                      (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs(), 0.), (0, 1, 0, 0))
                gy = F.pad(torch.where(valid[..., 1:, :] & valid[..., :-1, :],
                                      (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs(), 0.), (0, 0, 0, 1))
                strength = (gx + gy).sum(0)
                candidates = []
                for row in range(min(grid_size, height)):
                    y0, y1 = row * height // min(grid_size, height), (row + 1) * height // min(grid_size, height)
                    for col in range(min(grid_size, width)):
                        x0, x1 = col * width // min(grid_size, width), (col + 1) * width // min(grid_size, width)
                        coords = torch.nonzero(strength[y0:y1, x0:x1] > 0)
                        if coords.numel():
                            values = strength[y0:y1, x0:x1][coords[:, 0], coords[:, 1]]
                            count = max(1, math.ceil(len(values) * edge_fraction))
                            coords = coords[values.topk(count, sorted=False).indices]
                            candidates.append(coords + coords.new_tensor([y0, x0]))
                if candidates:
                    coords = candidates[torch.randint(len(candidates), ()).item()]
                    center = coords[torch.randint(len(coords), ()).item()]
                    top = max(0, min(height - crop_h, int(center[0]) - crop_h // 2))
                    left = max(0, min(width - crop_w, int(center[1]) - crop_w // 2))
            elif sampling != 'random':
                raise ValueError('Unknown local crop sampling mode.')
            def resize_patch(value):
                return F.interpolate(value[None, :, top:top + crop_h, left:left + crop_w],
                                     size=(height, width), mode='bilinear', align_corners=False)
            image_patches.append(resize_patch(image))
            targets.append(resize_patch(target))
        return torch.cat(image_patches), torch.cat(targets)

    @staticmethod
    def non_sky_ssi_loss(pred_depth, target_depth, sky_mask):
        """Per-image depth SSI; sky is excluded from statistics and residuals."""
        if pred_depth.shape != target_depth.shape or sky_mask.shape != target_depth.shape:
            raise ValueError('SSI inputs must share [B,C,H,W] dimensions.')
        target_depth = target_depth.detach()
        valid = ~sky_mask.detach().bool() & torch.isfinite(target_depth) & (target_depth > 0)
        image_losses = []
        for pred, target, mask in zip(pred_depth, target_depth, valid):
            if not mask.any():
                continue
            p, t = pred[mask], target[mask]
            pc, tc = p.median(), t.median()
            ps = (p - pc).abs().mean().clamp_min(1e-6)
            ts = (t - tc).abs().mean().clamp_min(1e-6)
            image_losses.append(((p - pc) / ps - (t - tc) / ts).abs().mean())
        return torch.stack(image_losses).mean() if image_losses else pred_depth.sum() * 0.

    def forward(self, inputs, outputs, student_model=None):
        """
        Calculates training supervised loss.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        """
        inv_depths = [outputs[("disp", 0, scale)] for scale in self.scales]
        with torch.no_grad():
            gt_inv_depth = forward_depth(self.teacher_net, inputs)
            gt_inv_depths = [gt_inv_depth[("disp", 0, scale)] for scale in self.scales]
        # Calculate supervised loss
        loss = self.calculate_loss(inv_depths, gt_inv_depths)
        self.last_full_loss = loss.detach()
        self.last_ssi_loss = loss.detach().new_zeros(())
        if self.training and self.ssi.ENABLED:
            target = inputs.get(('pseudo_metric', 0))
            sky = inputs.get(('pseudo_sky_mask', 0))
            if target is None or sky is None:
                raise ValueError('SSI requires pseudo_metric and pseudo_sky_mask inputs.')
            pred = outputs[('disp', 0, 0)].clamp_min(1e-6).reciprocal()
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            ssi = self.non_sky_ssi_loss(pred, target, sky)
            self.last_ssi_loss = ssi.detach()
            loss = loss + self.ssi.WEIGHT * ssi
        self.last_region_loss = loss.detach().new_zeros(())
        if self.training and self.region.ENABLED:
            region = self.region_abs_rel(outputs[('disp', 0, 0)], gt_inv_depth[('disp', 0, 0)],
                                         self.region.GRID_SIZE, self.region.FRACTION, self.region.RADIUS)
            self.last_region_loss = region.detach()
            loss = loss + self.region.WEIGHT * region
        self.last_boundary_loss = loss.detach().new_zeros(())
        self.last_boundary_weight = 0.0
        if self.training and self.boundary.ENABLED:
            relative = inputs.get(('pseudo_relative', 0))
            if relative is None:
                raise ValueError('Boundary supervision requires pseudo_relative input.')
            valid = torch.isfinite(relative) & (relative > self.boundary.RELATIVE_MIN)
            boundary = self.boundary_gradient_loss(outputs[('disp', 0, 0)], gt_inv_depth[('disp', 0, 0)], valid,
                                                   self.boundary.FRACTION, self.boundary.RADIUS)
            warmup = self.boundary.WARMUP_STEPS
            weight = self.boundary.WEIGHT * (min(1., self.optimization_step / warmup) if warmup else 1.)
            self.last_boundary_loss = boundary.detach()
            self.last_boundary_weight = weight
            loss = loss + weight * boundary
        self.last_hard_loss = loss.detach().new_zeros(())
        self.last_hard_weight = 0.0
        if self.training and self.hard_region.ENABLED:
            valid_mask = None
            if self.hard_region.EXCLUDE_RELATIVE_FAR:
                relative = inputs.get(('pseudo_relative', 0))
                if relative is None:
                    raise ValueError('Hard-region exclusion requires pseudo_relative input.')
                valid_mask = torch.isfinite(relative) & (relative > self.hard_region.RELATIVE_MIN)
            hard = self.hard_region_abs_rel(
                outputs[('disp', 0, 0)], gt_inv_depth[('disp', 0, 0)],
                self.hard_region.FRACTION, valid_mask)
            warmup = self.hard_region.WARMUP_STEPS
            weight = self.hard_region.WEIGHT * (min(1.0, self.optimization_step / warmup) if warmup else 1.0)
            self.last_hard_loss = hard.detach()
            self.last_hard_weight = weight
            loss = loss + weight * hard
        self.last_local_loss = loss.detach().new_zeros(())
        # Extra student pass only during training; teacher retains full-scene context.
        if self.training and self.local_crop.ENABLED:
            if student_model is None:
                raise ValueError('Local crop loss requires the student model.')
            if torch.rand(()).item() < self.local_crop.PROBABILITY:
                images, target = self.crop_pair(
                    inputs[('color_aug', 0)], gt_inv_depth[('disp', 0, 0)],
                    self.local_crop.MIN_FRACTION, self.local_crop.MAX_FRACTION,
                    self.local_crop.SAMPLING, self.local_crop.GRID_SIZE,
                    self.local_crop.EDGE_FRACTION)
                patches = {('color_aug', 0): images, 'weather_depth': inputs['weather_depth']}
                patch_outputs = forward_depth(student_model, patches, 'color_aug', 'weather_depth')
                # Full-resolution supervision avoids downsampling away small structures.
                pred = patch_outputs[('disp', 0, 0)]
                local = self.loss_func(pred, target)
                inv_min, inv_max = 1 / self.max_depth, 1 / self.min_depth
                metric = (pred.clamp(inv_min, inv_max).reciprocal()
                          - target.clamp(inv_min, inv_max).reciprocal()).abs().mean()
                if self.supervised_method == 'log_depth_a1':
                    local = local + self.metric_l1_weight * metric
                self.last_local_loss = local.detach()
                loss = loss + self.local_crop.WEIGHT * local
        # Return losses and metrics
        return loss
