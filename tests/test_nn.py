import torch
from torch.autograd import Variable, gradcheck
from tests.common import TestCase

from dsnt.nn import DSNT, dsnt, euclidean_loss, thresholded_softmax, kl_gauss_2d, mse_gauss_2d,\
    js_gauss_2d, make_gauss, variance_loss


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


class TestMakeGauss(TestCase):
    def test_make_gauss(self):
        expected = torch.Tensor([
            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
            [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
        ])
        actual = make_gauss(torch.Tensor([0, 0]), 5, 5, sigma=0.4)
        self.assertEqual(expected, actual, 1e-4)


def _test_reg_loss(tc, loss_method, shift_mean=True):
    # Target mean and standard deviation
    target_mean = torch.Tensor([0, 0])
    target_stddev = 0.4

    # Helper function to calculate the loss between the target and a Gaussian heatmap
    # parameterized by `mean` and `stddev`.
    def calc_loss(mean, stddev):
        hm = make_gauss(mean, 5, 5, sigma=stddev)
        return loss_method(hm, target_mean, sigma=target_stddev)

    # Minimum loss occurs when the heatmap's mean and standard deviation are the same
    # as the target
    min_loss = calc_loss(target_mean, target_stddev)

    # Minimum loss should be close to zero
    tc.assertEqual(min_loss, 0, 1e-3)

    # Loss should increase if the heatmap has a larger or smaller standard deviation than
    # the target
    tc.assertGreater(calc_loss(target_mean, target_stddev + 0.2), min_loss + 1e-3)
    tc.assertGreater(calc_loss(target_mean, target_stddev - 0.2), min_loss + 1e-3)

    if shift_mean:
        # Loss should increase if the heatmap has its mean location at a different
        # position than the target
        tc.assertGreater(calc_loss(target_mean + 0.1, target_stddev), min_loss + 1e-3)
        tc.assertGreater(calc_loss(target_mean - 0.1, target_stddev), min_loss + 1e-3)


class TestKLGaussLoss(TestCase):
    def test_kl_gauss_2d(self):
        _test_reg_loss(self, kl_gauss_2d)

    def test_mask(self):
        t = torch.Tensor([
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.1, 0.8],
            ],
            [
                [0.8, 0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ])
        coords = torch.Tensor([[1, 1], [0, 0]])
        mask = torch.Tensor([1, 0])

        actual = kl_gauss_2d(Variable(t), Variable(coords), Variable(mask), sigma=1)

        self.assertEqual(1.2228811717796824, actual.data[0])


class TestMSEGaussLoss(TestCase):
    def test_mse_gauss_2d(self):
        _test_reg_loss(self, mse_gauss_2d)


class TestJSGaussLoss(TestCase):
    def test_js_gauss_2d(self):
        _test_reg_loss(self, js_gauss_2d)


class TestVarianceLoss(TestCase):
    def test_variance_loss(self):
        def _variance_loss(inp, coords, mask=None, sigma=1):
            return variance_loss(inp, target_variance=sigma ** 2)

        _test_reg_loss(self, _variance_loss, shift_mean=False)
