from tests.common import TestCase

import torch
from torch.autograd import Variable
import torchvision.models

from dsnt.model import ResNetHumanPoseModel


class TestResNetHumanPoseModel(TestCase):
    def test_truncate(self):
        resnet = torchvision.models.resnet18()

        model = ResNetHumanPoseModel(resnet, n_chans=16, truncate=1)
        model.cuda()

        sz = model.image_specs.size
        self.assertEqual(sz, 224)
        out_var = model(Variable(torch.randn(1, 3, sz, sz).cuda()))
        self.assertEqual(out_var.size(), torch.Size([1, 16, 2]))

        hm = model.heatmaps.data
        self.assertEqual(hm.size(), torch.Size([1, 16, 14, 14]))

    def test_dilate(self):
        resnet = torchvision.models.resnet18()

        model = ResNetHumanPoseModel(resnet, n_chans=16, dilate=2)
        model.cuda()

        sz = model.image_specs.size
        self.assertEqual(sz, 224)
        out_var = model(Variable(torch.randn(1, 3, sz, sz).cuda()))
        self.assertEqual(out_var.size(), torch.Size([1, 16, 2]))

        hm = model.heatmaps.data
        self.assertEqual(hm.size(), torch.Size([1, 16, 28, 28]))

    def test_training_step(self):
        Tensor = torch.cuda.FloatTensor

        resnet = torchvision.models.resnet18()
        model = ResNetHumanPoseModel(resnet, n_chans=16, output_strat='dsnt', reg='js')
        model.type(Tensor)

        old_params = []
        for param in model.parameters():
            old_params.append(param.data.clone())

        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        in_var = Variable(Tensor(1, 3, 224, 224).uniform_(0, 1), requires_grad=False)
        target_var = Variable(Tensor(1, 16, 2).uniform_(-1, 1), requires_grad=False)

        out_var = model(in_var)
        loss = model.forward_loss(out_var, target_var, mask_var=None)
        loss.backward()

        optimizer.step()

        # Check that all parameter groups were updated
        for param_var, old_param in zip(model.parameters(), old_params):
            self.assertNotEqual(param_var.data, old_param)
