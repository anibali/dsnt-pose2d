import torch
from tests.common import TestCase

import dsnt.util

class TestUtil(TestCase):
    def test_draw_gaussian(self):
        expected = torch.FloatTensor([[
            [0.00000, 0.00000, 0.00005, 0.00020, 0.00034, 0.00020, 0.00005, 0.00000, 0.00000],
            [0.00000, 0.00012, 0.00150, 0.00674, 0.01111, 0.00674, 0.00150, 0.00012, 0.00000],
            [0.00005, 0.00150, 0.01832, 0.08208, 0.13534, 0.08208, 0.01832, 0.00150, 0.00005],
            [0.00020, 0.00674, 0.08208, 0.36788, 0.60653, 0.36788, 0.08208, 0.00674, 0.00020],
            [0.00034, 0.01111, 0.13534, 0.60653, 1.00000, 0.60653, 0.13534, 0.01111, 0.00034],
            [0.00020, 0.00674, 0.08208, 0.36788, 0.60653, 0.36788, 0.08208, 0.00674, 0.00020],
            [0.00005, 0.00150, 0.01832, 0.08208, 0.13534, 0.08208, 0.01832, 0.00150, 0.00005],
            [0.00000, 0.00012, 0.00150, 0.00674, 0.01111, 0.00674, 0.00150, 0.00012, 0.00000],
            [0.00000, 0.00000, 0.00005, 0.00020, 0.00034, 0.00020, 0.00005, 0.00000, 0.00000],
        ]])

        actual = torch.zeros(1, 9, 9).float()
        dsnt.util.draw_gaussian(actual, 4, 4, 1, normalize=False)

        self.assertEqual(expected, actual, 1e-5)

    def test_draw_gaussian_clipped(self):
        expected = torch.FloatTensor([[
            [0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
            [0.01111, 0.00674, 0.00150, 0.00012, 0.00000],
            [0.13534, 0.08208, 0.01832, 0.00150, 0.00000],
            [0.60653, 0.36788, 0.08208, 0.00674, 0.00000],
            [1.00000, 0.60653, 0.13534, 0.01111, 0.00000],
        ]])

        actual = torch.zeros(1, 5, 5).float()
        dsnt.util.draw_gaussian(actual, 0, 4, 1, normalize=False, clip_size=7)

        self.assertEqual(expected, actual, 1e-5)

    def test_encode_heatmaps(self):
        coords = torch.FloatTensor([[[-0.8, 0.8]]])

        expected = torch.FloatTensor([[[
            [0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
            [0.01111, 0.00674, 0.00150, 0.00012, 0.00000],
            [0.13534, 0.08208, 0.01832, 0.00150, 0.00000],
            [0.60653, 0.36788, 0.08208, 0.00674, 0.00000],
            [1.00000, 0.60653, 0.13534, 0.01111, 0.00000],
        ]]])

        actual = dsnt.util.encode_heatmaps(coords, 5, 5)

        self.assertEqual(expected, actual, 1e-5)

    def test_decode_heatmaps(self):
        heatmaps = torch.FloatTensor([[[
            [1.0, 0.0],
            [0.0, 0.0],
        ]]])

        expected = torch.FloatTensor([[[-0.5, -0.5]]])
        actual = dsnt.util.decode_heatmaps(heatmaps)

        self.assertEqual(expected, actual, 1e-7)
