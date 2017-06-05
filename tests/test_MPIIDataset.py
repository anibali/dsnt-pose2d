import unittest
from common import TestCase

import torch

from dsnt.data import MPIIDataset

class TestMPIIDataset(TestCase):
  def test_len(self):
    test_set = MPIIDataset('/data/dlds/mpii-human-pose', 'test')
    test_set_len = 11731
    self.assertEqual(test_set_len, len(test_set))

    train_set = MPIIDataset('/data/dlds/mpii-human-pose', 'train')
    train_set_len = 25925
    self.assertEqual(train_set_len, len(train_set))

  def test_getitem(self):
    dataset = MPIIDataset('/data/dlds/mpii-human-pose', 'train', use_aug=False)
    sample = dataset[543]
    self.assertIn('input', sample)
    self.assertIn('part_coords', sample)
