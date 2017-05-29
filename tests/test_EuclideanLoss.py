import unittest
import torch
from torch.autograd import Variable
from common import TestCase

from dsnt.nn import EuclideanLoss

class TestEuclideanLoss(TestCase):
  def test_forward_and_backward(self):
    criterion = EuclideanLoss()

    input = torch.Tensor([
      [[3, 4], [3, 4]],
      [[3, 4], [3, 4]],
    ])

    target = torch.Tensor([
      [[0, 0], [0, 0]],
      [[0, 0], [0, 0]],
    ])

    in_var = Variable(input, requires_grad=True)

    expected_loss = torch.Tensor([5])
    actual_loss = criterion.forward(in_var, Variable(target))
    expected_grad = torch.Tensor([
      [[0.15, 0.20], [0.15, 0.20]],
      [[0.15, 0.20], [0.15, 0.20]],
    ])
    actual_loss.backward()
    actual_grad = in_var.grad.data

    self.assertEqual(expected_loss, actual_loss.data)
    self.assertEqual(expected_grad, actual_grad)

  def test_mask(self):
    criterion = EuclideanLoss()

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
    actual = criterion(Variable(output), Variable(target), Variable(mask))

    self.assertEqual(expected, actual.data)
