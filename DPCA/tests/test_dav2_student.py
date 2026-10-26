import unittest
from unittest.mock import patch

import torch
from torch import nn

from config.config import get_cfg
from models.depth_factory import build_depth_model, forward_depth


class FakeMetricNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pretrained = nn.Conv2d(3, 1, 1)
        self.depth_head = nn.Conv2d(1, 1, 1)
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach()
        return 1 + self.depth_head(self.pretrained(x)).sigmoid()[:, 0] * 20


class DAV2StudentTests(unittest.TestCase):
    def make_student(self):
        cfg = get_cfg()
        cfg.merge_from_file('config/train_md4allDD_dav2_small_robotcar.yaml')
        with patch('models.dav2_student.metric_model_class', return_value=FakeMetricNet):
            model = build_depth_model(cfg, initialize_pretrained=False)
        return cfg, model

    def test_rgb_only_alignment_normalization_and_backward(self):
        cfg, model = self.make_student()
        rgb = torch.rand(2, 3, 32, 54)
        outputs = forward_depth(model, {('color', 0): rgb})
        x = model.network.last_input
        self.assertEqual(x.shape[-2:], (42, 56))
        torch.testing.assert_close(x[:, :, :32, :54], (rgb - model.image_mean) / model.image_std)
        torch.testing.assert_close(x[:, :, 32:, :54], x[:, :, 31:32, :54].expand(-1, -1, 10, -1))
        for scale in cfg.DATASET.SCALES:
            pred = outputs[('disp', 0, scale)]
            self.assertEqual(pred.shape, (2, 1, 32 // 2 ** scale, 54 // 2 ** scale))
            self.assertTrue(torch.isfinite(pred).all())
            self.assertTrue((pred > 0).all())
        outputs[('disp', 0, 0)].log().mean().backward()
        for module in (model.network.pretrained, model.network.depth_head):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)

    def test_lr_groups_cover_student_and_restore_without_pretrain(self):
        cfg, model = self.make_student()
        groups = model.optimizer_param_groups(cfg.OPTIMIZER.DEPTH_LR, cfg.MODEL.DEPTH.DAV2.ENCODER_LR_FACTOR)
        self.assertAlmostEqual(groups[0]['lr'], 1e-5)
        self.assertAlmostEqual(groups[1]['lr'], 1e-4)
        params = [p for group in groups for p in group['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        _, restored = self.make_student()
        restored.load_state_dict(model.state_dict(), strict=True)
        image = torch.rand(1, 3, 32, 54)
        torch.testing.assert_close(restored(image)[('disp', 0, 0)], model(image)[('disp', 0, 0)])

    def test_missing_pretraining_fails_instead_of_training_randomly(self):
        cfg = get_cfg()
        cfg.MODEL.DEPTH.TYPE = 'dav2_small'
        cfg.MODEL.DEPTH.DAV2.PRETRAINED_PATH = '/tmp/missing-dav2-weights-for-test.pth'
        with patch('models.dav2_student.metric_model_class', return_value=FakeMetricNet):
            with self.assertRaises(FileNotFoundError):
                build_depth_model(cfg)


if __name__ == '__main__':
    unittest.main()
