#!/usr/bin/env python3

"""
This script is for investigating misclassified examples.
"""

import argparse
import random

import torch
import h5py
import numpy as np

from dsnt.data import MPIIDataset


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='Prediction investigator')
    parser.add_argument('--preds', type=str, metavar='PATH', required=True,
                        help='predictions file')
    parser.add_argument('--subset', type=str, default='val', metavar='S',
                        help='data subset to evaluate on (default="val")')
    parser.add_argument('--seed', type=int, metavar='N',
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


def main():
    """Main evaluation entrypoint function."""

    args = parse_args()
    seed_random_number_generators(args.seed)

    preds_file = args.preds
    subset = args.subset

    dataset = MPIIDataset('/data/dlds/mpii-human-pose', subset, use_aug=False)
    dataset._skip_read_image = True

    # Load predictions from file
    with h5py.File(preds_file, 'r') as f:
        preds = torch.from_numpy(f['preds'][:]).double()

    n_joints = 16
    threshold = 0.5

    actual_file = '/data/dlds/mpii-human-pose/annot-{}.h5'.format(dataset.subset)
    with h5py.File(actual_file, 'r') as f:
        actual = torch.from_numpy(f['part'][:])
        head_lengths = torch.from_numpy(f['normalize'][:])
    joint_mask = actual.select(2, 0).gt(1).mul(actual.select(2, 1).gt(1))

    total_misclassified = 0
    edge_misclassified = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        norm_targets = sample['part_coords']

        for j in range(n_joints):
            if joint_mask[i, j] == 1:
                dist = torch.dist(actual[i, j], preds[i, j]) / head_lengths[i]
                if dist > threshold:
                    total_misclassified += 1
                    if norm_targets[j].abs().max() > 0.9:
                        edge_misclassified += 1

    print('total_misclassified', total_misclassified)
    print('edge_misclassified: {} ({:.2f}%)'.format(edge_misclassified, 100.0 * edge_misclassified / total_misclassified))


if __name__ == '__main__':
    main()
