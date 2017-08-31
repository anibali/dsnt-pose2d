import torch
from torch.autograd import Variable
from tests.common import TestCase

from dsnt.nn import DSNT

SIMPLE_INPUT = torch.Tensor([[[
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1, 0.6, 0.1],
    [0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
]]])

SIMPLE_OUTPUT = torch.Tensor([[[0.4, 0.0]]])

SIMPLE_TARGET = torch.Tensor([[[0.5, 0.5]]])

# Expected dloss/dinput when using MSE with target (0.5, 0.5)
SIMPLE_GRAD_INPUT = torch.Tensor([[[
    [0.4800, 0.4400, 0.4000, 0.3600, 0.3200],
    [0.2800, 0.2400, 0.2000, 0.1600, 0.1200],
    [0.0800, 0.0400, 0.0000, -0.0400, -0.0800],
    [-0.1200, -0.1600, -0.2000, -0.2400, -0.2800],
    [-0.3200, -0.3600, -0.4000, -0.4400, -0.4800],
]]])

class TestDSNT(TestCase):
    def test_forward(self):
        layer = DSNT()
        in_var = Variable(SIMPLE_INPUT, requires_grad=False)

        expected = SIMPLE_OUTPUT
        actual = layer(in_var)
        self.assertEqual(actual.data, expected)

    def test_backward(self):
        layer = DSNT()
        mse = torch.nn.MSELoss()

        in_var = Variable(SIMPLE_INPUT, requires_grad=True)
        output = layer(in_var)

        target_var = Variable(SIMPLE_TARGET, requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected = SIMPLE_GRAD_INPUT
        actual = in_var.grad.data
        self.assertEqual(actual, expected)

    def test_batchless(self):
        layer = DSNT()
        mse = torch.nn.MSELoss()

        in_var = Variable(SIMPLE_INPUT.squeeze(0), requires_grad=True)

        expected_output = SIMPLE_OUTPUT.squeeze(0)
        output = layer(in_var)
        self.assertEqual(output.data, expected_output)

        target_var = Variable(SIMPLE_TARGET.squeeze(0), requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected_grad = SIMPLE_GRAD_INPUT.squeeze(0)
        self.assertEqual(in_var.grad.data, expected_grad)

    def test_cuda(self):
        layer = DSNT()
        mse = torch.nn.MSELoss()

        in_var = Variable(SIMPLE_INPUT.cuda(), requires_grad=True)

        expected_output = SIMPLE_OUTPUT.cuda()
        output = layer(in_var)
        self.assertEqual(output.data, expected_output)

        target_var = Variable(SIMPLE_TARGET.cuda(), requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected_grad = SIMPLE_GRAD_INPUT.cuda()
        self.assertEqual(in_var.grad.data, expected_grad)
