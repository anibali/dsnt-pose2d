import torch
from tests.common import TestCase

from dsnt.evaluator import PCKhEvaluator


class TestPCKhEvaluator(TestCase):
    def test_calculate_pckh_distance(self):
        pred = torch.Tensor([951.84, 580.64])
        target = torch.Tensor([804, 711])
        head_length = 117.962

        expected = 1.6709
        actual = PCKhEvaluator.calculate_pckh_distance(pred, target, head_length)

        self.assertEqual(actual, expected, 1e-4)

    def test_add(self):
        evaluator = PCKhEvaluator(threshold=0.5)

        pred = torch.Tensor([
            [[951.84, 580.64]],
            [[317.76, 406.75]],
            [[float('inf'), float('inf')]],
        ])
        target = torch.Tensor([
            [[804, 711]],
            [[317, 412]],
            [[float('nan'), float('nan')]],
        ])
        head_length = torch.Tensor([117.962, 44.046, 78.481])
        joint_mask = torch.Tensor([[1], [1], [0]])

        evaluator.add(pred, target, joint_mask, head_length)

        expected = 0.5
        actual, _ = evaluator.meters['all'].value()

        self.assertEqual(actual, expected)
