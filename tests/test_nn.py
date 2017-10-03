import torch
from torch.autograd import Variable, gradcheck
from tests.common import TestCase

from dsnt.nn import DSNT, dsnt, euclidean_loss, thresholded_softmax


class TestDSNT(TestCase):
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

    def test_forward(self):
        layer = DSNT()
        in_var = Variable(self.SIMPLE_INPUT, requires_grad=False)

        expected = self.SIMPLE_OUTPUT
        actual = layer(in_var)
        self.assertEqual(actual.data, expected)

    def test_backward(self):
        layer = DSNT()
        mse = torch.nn.MSELoss()

        in_var = Variable(self.SIMPLE_INPUT, requires_grad=True)
        output = layer(in_var)

        target_var = Variable(self.SIMPLE_TARGET, requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected = self.SIMPLE_GRAD_INPUT
        actual = in_var.grad.data
        self.assertEqual(actual, expected)

    def test_batchless(self):
        layer = DSNT()
        mse = torch.nn.MSELoss()

        in_var = Variable(self.SIMPLE_INPUT.squeeze(0), requires_grad=True)

        expected_output = self.SIMPLE_OUTPUT.squeeze(0)
        output = layer(in_var)
        self.assertEqual(output.data, expected_output)

        target_var = Variable(self.SIMPLE_TARGET.squeeze(0), requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected_grad = self.SIMPLE_GRAD_INPUT.squeeze(0)
        self.assertEqual(in_var.grad.data, expected_grad)

    def test_cuda(self):
        layer = DSNT()
        mse = torch.nn.MSELoss()

        in_var = Variable(self.SIMPLE_INPUT.cuda(), requires_grad=True)

        expected_output = self.SIMPLE_OUTPUT.cuda()
        output = layer(in_var)
        self.assertEqual(output.data, expected_output)

        target_var = Variable(self.SIMPLE_TARGET.cuda(), requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected_grad = self.SIMPLE_GRAD_INPUT.cuda()
        self.assertEqual(in_var.grad.data, expected_grad)


class TestFunctionalDSNT(TestCase):
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

    def test_forward(self):
        in_var = Variable(self.SIMPLE_INPUT, requires_grad=False)

        expected = self.SIMPLE_OUTPUT
        actual = dsnt(in_var)
        self.assertEqual(actual.data, expected)

    def test_backward(self):
        mse = torch.nn.MSELoss()

        in_var = Variable(self.SIMPLE_INPUT, requires_grad=True)
        output = dsnt(in_var)

        target_var = Variable(self.SIMPLE_TARGET, requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected = self.SIMPLE_GRAD_INPUT
        actual = in_var.grad.data
        self.assertEqual(actual, expected)

    def test_batchless(self):
        mse = torch.nn.MSELoss()

        in_var = Variable(self.SIMPLE_INPUT.squeeze(0), requires_grad=True)

        expected_output = self.SIMPLE_OUTPUT.squeeze(0)
        output = dsnt(in_var)
        self.assertEqual(output.data, expected_output)

        target_var = Variable(self.SIMPLE_TARGET.squeeze(0), requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected_grad = self.SIMPLE_GRAD_INPUT.squeeze(0)
        self.assertEqual(in_var.grad.data, expected_grad)

    def test_cuda(self):
        mse = torch.nn.MSELoss()

        in_var = Variable(self.SIMPLE_INPUT.cuda(), requires_grad=True)

        expected_output = self.SIMPLE_OUTPUT.cuda()
        output = dsnt(in_var)
        self.assertEqual(output.data, expected_output)

        target_var = Variable(self.SIMPLE_TARGET.cuda(), requires_grad=False)
        loss = mse(output, target_var)
        loss.backward()

        expected_grad = self.SIMPLE_GRAD_INPUT.cuda()
        self.assertEqual(in_var.grad.data, expected_grad)

    def test_square_coords(self):
        in_var = Variable(torch.Tensor([[
            [0.0, 0.1, 0.0],
            [0.1, 0.6, 0.1],
            [0.0, 0.1, 0.0],
        ]]))

        expected = torch.Tensor([[2 * 0.1 * (2/3)**2, 2 * 0.1 * (2/3)**2]])
        out_var = dsnt(in_var, square_coords=True)

        self.assertEqual(out_var.data, expected)


class TestEuclideanLoss(TestCase):
    def test_forward_and_backward(self):
        input_tensor = torch.Tensor([
            [[3, 4], [3, 4]],
            [[3, 4], [3, 4]],
        ])

        target = torch.Tensor([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ])

        in_var = Variable(input_tensor, requires_grad=True)

        expected_loss = torch.Tensor([5])
        actual_loss = euclidean_loss(in_var, Variable(target))
        expected_grad = torch.Tensor([
            [[0.15, 0.20], [0.15, 0.20]],
            [[0.15, 0.20], [0.15, 0.20]],
        ])
        actual_loss.backward()
        actual_grad = in_var.grad.data

        self.assertEqual(expected_loss, actual_loss.data)
        self.assertEqual(expected_grad, actual_grad)

    def test_mask(self):
        output = torch.Tensor([
            [[0, 0], [1, 1], [0, 0]],
            [[1, 1], [0, 0], [0, 0]],
        ])

        target = torch.Tensor([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ])

        mask = torch.Tensor([
            [1, 0, 1],
            [0, 1, 1],
        ])

        expected = torch.Tensor([0])
        actual = euclidean_loss(Variable(output), Variable(target), Variable(mask))

        self.assertEqual(expected, actual.data)


class TestThresholdedSoftmax(TestCase):
    def test_forward(self):
        in_var = Variable(torch.Tensor([2, 1, 3]))
        actual = thresholded_softmax(in_var, 1.5).data
        expected = torch.Tensor([0.26894142, 0, 0.73105858])
        self.assertEqual(actual, expected)

    def test_backward(self):
        in_var = Variable(torch.randn(20), requires_grad=True)
        threshold = 0
        self.assertTrue(gradcheck(thresholded_softmax, (in_var, threshold)))

    def test_forward_batch(self):
        in_var = Variable(torch.Tensor([[2, 1, 3], [4, 0, 0]]))
        actual = thresholded_softmax(in_var, 1.5).data
        expected = torch.Tensor([[0.26894142, 0, 0.73105858], [1, 0, 0]])
        self.assertEqual(actual, expected)

    def test_backward_batch(self):
        in_var = Variable(torch.randn(3, 20), requires_grad=True)
        threshold = 0
        self.assertTrue(gradcheck(thresholded_softmax, (in_var, threshold)))
