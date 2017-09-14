from tests.common import TestCase

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.models

from dsnt.model import ResNetHumanPoseModel

class TestResNetHumanPoseModel(TestCase):
    def test_truncate(self):
        resnet = torchvision.models.resnet18()

        model = ResNetHumanPoseModel(resnet, n_chans=16, truncate=1)

        sz = model.image_specs.size
        self.assertEqual(sz, 224)
        out_var = model(Variable(torch.randn(1, 3, sz, sz)))
        self.assertEqual(out_var.size(), torch.Size([1, 16, 2]))

        hm = model.heatmaps.data
        self.assertEqual(hm.size(), torch.Size([1, 16, 14, 14]))

    def test_dilate(self):
        resnet = torchvision.models.resnet18()

        model = ResNetHumanPoseModel(resnet, n_chans=16, dilate=2)

        sz = model.image_specs.size
        self.assertEqual(sz, 224)
        out_var = model(Variable(torch.randn(1, 3, sz, sz)))
        self.assertEqual(out_var.size(), torch.Size([1, 16, 2]))

        hm = model.heatmaps.data
        self.assertEqual(hm.size(), torch.Size([1, 16, 28, 28]))
