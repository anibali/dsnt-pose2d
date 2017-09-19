#!/usr/bin/env python3

"""
This script will benchmark a trained model.

The model will be measured for speed and accuracy on the validation set.
"""

import argparse
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import h5py
import numpy as np
from scipy.stats import norm
import progressbar
from torchnet.meter.meter import Meter

from dsnt.model import build_mpii_pose_model
from dsnt.data import MPIIDataset
from dsnt.evaluator import PCKhEvaluator
from dsnt.util import timer


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model evaluator')
    parser.add_argument(
        '--model', type=str, metavar='PATH', required=True,
        help='model state file')
    parser.add_argument(
        '--seed', type=int, metavar='N',
        help='seed for random number generators')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 999999)

    return args


def seed_random_number_generators(seed):
    """Seed all random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MedianValueMeter(Meter):
    def __init__(self):
        super().__init__()
        self.values = []
        self.reset()

    def add(self, value):
        self.values.append(value)

    def value(self):
        data = np.asarray(self.values)

        median = np.median(data)

        k = norm.ppf(0.75) # Assume data is normally distributed
        mad = np.median(np.fabs(data - median) / k)

        return median, mad

    def reset(self):
        self.values.clear()


def main():
    """Main benchmark entrypoint function."""

    args = parse_args()
    seed_random_number_generators(args.seed)

    model_file = args.model

    batch_size = 1
    subset = 'val'

    # Ground truth
    actual_file = '/data/dlds/mpii-human-pose/annot-' + subset + '.h5'
    with h5py.File(actual_file, 'r') as f:
        actual = torch.from_numpy(f['part'][:])
        head_lengths = torch.from_numpy(f['normalize'][:])
    joint_mask = actual.select(2, 0).gt(1).mul(actual.select(2, 1).gt(1))

    # Generate predictions with the model

    model_state = torch.load(model_file)

    print(model_state['model_desc'])

    model = build_mpii_pose_model(**model_state['model_desc'])
    model.load_state_dict(model_state['state_dict'])
    model.cuda()
    model.eval()

    print('Number of parameters: {:0.2f} million'.format(
        sum(p.numel() for p in model.parameters()) * 1e-6))

    dataset = MPIIDataset(
        '/data/dlds/mpii-human-pose', subset, use_aug=False, image_specs=model.image_specs)
    loader = DataLoader(dataset, batch_size, num_workers=4, pin_memory=True)
    preds = torch.DoubleTensor(len(dataset), 16, 2)

    inference_time_meter = MedianValueMeter()

    completed = 0
    with progressbar.ProgressBar(max_value=len(dataset)) as bar:
        for i, batch in enumerate(loader):
            in_var = Variable(batch['input'].cuda(), volatile=True)

            with timer(inference_time_meter):
                out_var = model(in_var)
                coords = model.compute_coords(out_var)

            norm_preds = coords.double()
            pos = i * batch_size
            orig_preds = preds[pos:(pos + norm_preds.size(0))]
            torch.baddbmm(
                batch['transform_b'],
                norm_preds,
                batch['transform_m'],
                out=orig_preds)

            completed += in_var.data.size(0)
            bar.update(completed)

    # PyCharm console output breaks unless we pause here briefly
    time.sleep(0.2)

    time_mean, time_std = inference_time_meter.value()
    print('Inference time: {:0.2f}±{:0.2f} ms'.format(time_mean * 1000, time_std * 1000))

    # Calculate PCKh accuracies
    evaluator = PCKhEvaluator()
    evaluator.add(preds, actual, joint_mask, head_lengths)

    mean_pckh, _ = evaluator.meters['all'].value()
    print('Mean PCKh: {:0.6f}'.format(mean_pckh))
    mean_hard_pckh, _ = evaluator.meters['all_hard'].value()
    print('Mean hard PCKh: {:0.6f}'.format(mean_hard_pckh))


if __name__ == '__main__':
    main()
