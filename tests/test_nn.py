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


class TestKLGaussLoss(TestCase):
    def test_kl_gauss_2d(self):
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
        coords = torch.Tensor([[1, 1], [-1, -1]])

        self.assertEqual(1.2228811717796824, kl_gauss_2d(t, coords, sigma=1))
        self.assertEqual(1.2228811717796824, kl_gauss_2d(t[0], coords[0], sigma=1))

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
        t = torch.Tensor([
            [
                [0.0081, 0.0172, 0.0284, 0.0364],
                [0.0172, 0.0364, 0.0601, 0.0772],
                [0.0284, 0.0601, 0.0991, 0.1272],
                [0.0364, 0.0772, 0.1272, 0.1633],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.1, 0.2],
                [0.0, 0.1, 0.2, 0.3],
            ]
        ])
        coords = torch.Tensor([[1, 1], [1, 1]])

        self.assertEqual(0.021896807803733806, mse_gauss_2d(t, coords, sigma=1))
        self.assertEqual(0, mse_gauss_2d(t[0], coords[0], sigma=1))


class TestJSGaussLoss(TestCase):
    def test_js_gauss_2d(self):
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
        coords = torch.Tensor([[1, 1], [-1, -1]])

        self.assertEqual(0.3180417843094644, js_gauss_2d(t, coords, sigma=1))
        self.assertEqual(0.3180417843094644, js_gauss_2d(t[0], coords[0], sigma=1))


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


class TestVarianceLoss(TestCase):
    def test_variance_loss(self):
        hm = torch.Tensor([
            [
                [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
            ],
        ])

        self.assertEqual(variance_loss(hm, target_variance=0.4**2), 0, 1e-3)
        self.assertGreater(variance_loss(hm, target_variance=0.5**2), 1e-2)
        self.assertGreater(variance_loss(hm, target_variance=0.2**2), 1e-2)
