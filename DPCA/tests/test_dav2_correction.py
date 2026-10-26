import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config.config import get_cfg
from data.pseudo_depth import PseudoDepthReader
from data.transforms import Crop, Resize, DaytimeTranslation
from losses.MultiViewPhotometricLoss import MultiViewPhotometricLoss
from losses.SupervisedLoss import SupervisedLoss
from losses.VelocityLoss import VelocityLoss
from models.depth_correction import DepthCorrectionNet
from models.depth_factory import forward_depth
from utils.camera import Warping
from utils.pose import Pose
from torchvision.transforms import InterpolationMode


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.cfg = get_cfg()
        self.cfg.MODEL.DEPTH.TYPE = 'dav2_correction'
        self.cfg.MODEL.DEPTH.CORRECTION.CHANNELS = 8
        self.cfg.DATASET.SCALES = [0, 1, 2, 3]

    def batch(self):
        return {('color', 0): torch.rand(2, 3, 32, 64),
                ('pseudo_metric', 0): 2 + 30 * torch.rand(2, 1, 32, 64),
                ('pseudo_relative', 0): torch.rand(2, 1, 32, 64),
                'weather': ['day-clear', 'day-clear']}

    def test_identity_multiscale_and_gradients(self):
        model = DepthCorrectionNet(self.cfg)
        batch = self.batch()
        batch[('pseudo_metric', 0)].requires_grad_()
        batch[('pseudo_relative', 0)].requires_grad_()
        outputs = forward_depth(model, batch)
        torch.testing.assert_close(outputs[('disp', 0, 0)].reciprocal(), batch[('pseudo_metric', 0)])
        for scale in self.cfg.DATASET.SCALES:
            self.assertEqual(outputs[('disp', 0, scale)].shape, (2, 1, 32 // 2 ** scale, 64 // 2 ** scale))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            pred = forward_depth(model, batch)[('disp', 0, 0)]
            (pred - 1 / (batch[('pseudo_metric', 0)].detach() * 0.7)).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        self.assertGreater(model.enc0[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(batch[('pseudo_metric', 0)].grad)
        self.assertIsNone(batch[('pseudo_relative', 0)].grad)

    def test_flip_keeps_all_inputs_aligned(self):
        model = DepthCorrectionNet(self.cfg)
        batch = self.batch()
        inputs_seen = []
        handle = model.enc0.register_forward_pre_hook(lambda _, inputs: inputs_seen.append(inputs[0].clone()))
        forward_depth(model, batch)
        flipped = forward_depth(model, batch, flip=True)
        handle.remove()
        torch.testing.assert_close(inputs_seen[0].flip(-1), inputs_seen[1])
        torch.testing.assert_close(flipped[('disp', 0, 0)].flip(-1).reciprocal(), batch[('pseudo_metric', 0)])

    def test_png_decode_crop_and_resize(self):
        cfg = self.cfg
        cfg.DATASET.AUGMENTATION.CROP.TOP = 1
        cfg.DATASET.AUGMENTATION.CROP.LEFT = 2
        cfg.DATASET.AUGMENTATION.CROP.HEIGHT = 2
        cfg.DATASET.AUGMENTATION.CROP.WIDTH = 4
        cfg.DATASET.AUGMENTATION.RESIZE.HEIGHT = 2
        cfg.DATASET.AUGMENTATION.RESIZE.WIDTH = 4
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('metric', 'relative'):
                root = Path(tmp) / kind
                root.mkdir()
                setattr(cfg.DATASET.PSEUDO_DEPTH, kind.upper() + '_PATH', str(root))
                manifest = dict(geometry='original_uncropped_image', png_scale_m=80.0)
                if kind == 'metric':
                    manifest.update(format='metric_depth_uint16_png_fixed_scale_v1', representation='metric_depth')
                else:
                    manifest.update(format='relative_inverse_depth_uint16_png_max_per_image_v1', representation='relative_inverse_depth')
                (root / 'manifest.json').write_text(json.dumps(manifest))
                array = np.arange(1, 25, dtype=np.uint16).reshape(4, 6) * 1000
                Image.fromarray(array).save(root / 'frame.png.png')
            reader = PseudoDepthReader(cfg)
            result = reader('frame.png', (4, 6))
            expected = torch.tensor(array[1:3, 2:6].astype(np.float32)).unsqueeze(0) / 65535
            torch.testing.assert_close(result[('pseudo_relative', 0)], expected)
            torch.testing.assert_close(result[('pseudo_metric', 0)], expected * 80)
            with self.assertRaises(ValueError):
                reader('frame.png', (8, 12))
            with self.assertRaises(FileNotFoundError):
                reader('missing.png', (4, 6))

    def test_intrinsics_crop_and_pixel_center_resize(self):
        K = torch.tensor([[100., 0., 60.], [0., 120., 50.], [0., 0., 1.]])
        sample = {'K': K.clone()}
        sample = Crop(10, 20, 80, 100, ['K'], [0])(sample)
        sample = Resize((40, 25), InterpolationMode.BILINEAR, True, ['K'], [0],
                        (80, 100), pixel_center_intrinsics=True)(sample)
        expected = torch.tensor([[25., 0., 9.625], [0., 60., 19.75], [0., 0., 1.]])
        torch.testing.assert_close(sample['K'], expected)

    def test_translated_student_image_matches_training_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new('RGB', (544, 320), (10, 20, 30)).save(Path(tmp) / 'frame.png')
            sample = {'weather': 'day-clear', ('filename', 0): 'frame.png',
                      ('color_aug', 0): Image.new('RGB', (128, 64))}
            sample = DaytimeTranslation(tmp, 'day->night', 1., ['color_aug'], [0])(sample)
            self.assertEqual(sample[('color_aug', 0)].size, (128, 64))
            self.assertIn('translated', sample['weather_depth'])

    def test_metric_scale_ambiguity_and_odometry(self):
        cfg = self.cfg
        cfg.DATASET.TEMP_CONTEXT = [0, 1]
        K = torch.tensor([[[30., 0., 16.], [0., 30., 8.], [0., 0., 1.]]])
        depth = torch.ones(1, 1, 16, 32) * 10
        rgb = torch.rand(1, 3, 16, 32)
        pose = Pose.from_vec(torch.tensor([[0.2, 0., 0., 0., 0., 0.]]), 'euler')
        scaled = Pose(pose.mat.clone())
        scaled.mat[:, :3, 3] *= 2
        warp = Warping([0])
        image1, _, coords = warp.warp_ref_image([1 / depth], rgb, K, K, pose)
        image2, _, _ = warp.warp_ref_image([1 / (2 * depth)], rgb, K, K, scaled)
        torch.testing.assert_close(image1[0], image2[0])
        # Positive target->source x translation shifts sampling coordinates right.
        self.assertAlmostEqual(coords[0][0, 8, 16, 0].item(), 2 * (16 + .6) / 31 - 1, places=5)
        loss = VelocityLoss(cfg)
        inputs = {('pose_gt', 1): pose.mat}
        self.assertEqual(loss(inputs, {('pose', 1): pose}).item(), 0)
        self.assertAlmostEqual(loss(inputs, {('pose', 1): scaled}).item(), .2, places=5)

    def test_invalid_projection_has_finite_zero_loss(self):
        cfg = self.cfg
        cfg.DATASET.SCALES = [0]
        cfg.DATASET.TEMP_CONTEXT = [0, 1]
        cfg.LOSS.PHOTOMETRIC.MASK_INVALID = True
        cfg.DATASET.AUGMENTATION.RESIZE.HEIGHT = 16
        cfg.DATASET.AUGMENTATION.RESIZE.WIDTH = 32
        inputs = {('color', 0): torch.rand(1, 3, 16, 32),
                  ('color', 1): torch.rand(1, 3, 16, 32),
                  'K': torch.eye(3).unsqueeze(0)}
        for ssim in (0., .85):
            cfg.LOSS.PHOTOMETRIC.SSIM.WEIGHT = ssim
            for z in (-10., -11.):  # Exactly zero projected depth, or behind the camera.
                inv = torch.full((1, 1, 16, 32), .1, requires_grad=True)
                pose = Pose.from_vec(torch.tensor([[0., 0., z, 0., 0., 0.]]), 'euler')
                loss = MultiViewPhotometricLoss(cfg)(inputs, {('disp', 0, 0): inv, ('pose', 1): pose})
                self.assertEqual(loss.item(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(inv.grad).all())

    def test_photometric_loss_reaches_correction_and_pose(self):
        cfg = self.cfg
        cfg.DATASET.SCALES = [0]
        cfg.DATASET.TEMP_CONTEXT = [0, 1]
        cfg.LOSS.PHOTOMETRIC.MASK_INVALID = True
        cfg.LOSS.PHOTOMETRIC.AUTOMASK = False
        model = DepthCorrectionNet(cfg)
        batch = self.batch()
        batch[('color', 1)] = torch.rand_like(batch[('color', 0)])
        batch['K'] = torch.tensor([[[30., 0., 32.], [0., 30., 16.], [0., 0., 1.]]]).repeat(2, 1, 1)
        pose_vec = torch.tensor([[.1, 0., 0., 0., 0., 0.]]).repeat(2, 1).requires_grad_()
        outputs = forward_depth(model, batch)
        outputs[('pose', 1)] = Pose.from_vec(pose_vec, 'euler')
        loss = MultiViewPhotometricLoss(cfg)(batch, outputs)
        loss.backward()
        self.assertGreater(model.residual.weight.grad.abs().sum().item(), 0)
        self.assertGreater(pose_vec.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.residual.weight.grad).all())
        self.assertTrue(torch.isfinite(pose_vec.grad).all())

    def test_checkpoint_teacher_is_frozen_and_uses_correction(self):
        cfg = self.cfg
        teacher = DepthCorrectionNet(cfg)
        cfg.LOSS.SUPERVISED.TEACHER_TYPE = 'dav2_correction'
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'teacher.ckpt')
            torch.save({'state_dict': {'depth_model.' + k: v for k, v in teacher.state_dict().items()}}, path)
            cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATH = path
            supervised = SupervisedLoss(cfg, True)
            supervised.train()
            self.assertFalse(supervised.teacher_net.training)
            self.assertTrue(all(not p.requires_grad for p in supervised.teacher_net.parameters()))
            batch = self.batch()
            output = forward_depth(teacher, batch)
            self.assertLess(supervised(batch, output).item(), 1e-7)


if __name__ == '__main__':
    unittest.main()
