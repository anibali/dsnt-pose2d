from tests.common import TestCase

from dsnt.data import MPIIDataset, ImageSpecs


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

    def test_image_specs(self):
        image_specs = ImageSpecs(size=128, subtract_mean=True, divide_stddev=False)
        dataset = MPIIDataset(
            '/data/dlds/mpii-human-pose', 'train', use_aug=False, image_specs=image_specs)
        sample = dataset[42]

        self.assertEqual((3, 128, 128), sample['input'].size())
        self.assertEqual(-0.444027, sample['input'].min())
        self.assertEqual(0.567317, sample['input'].max())
