#!/usr/bin/env python3

"""
This script is for investigating misclassified examples.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette

import argparse
import random

import h5py
import numpy as np
import torch
from scipy.stats import binned_statistic_2d
from scipy.signal import convolve

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
    """Main investigation entrypoint function."""

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

    true_xs = []
    true_ys = []
    pred_xs = []
    pred_ys = []

    txs = []
    tys = []

    for i in range(len(dataset)):
        sample = dataset[i]
        norm_targets = sample['part_coords']
        norm_preds = torch.mm((preds[i] - sample['transform_b']).squeeze(), torch.inverse(sample['transform_m']))

        for j in range(n_joints):
            if joint_mask[i, j] == 1 and np.abs(norm_targets[j, 0]) <= 1 and np.abs(norm_targets[j, 1]) <= 1:
                txs.append(norm_targets[j, 0])
                tys.append(norm_targets[j, 1])
                dist = torch.dist(actual[i, j], preds[i, j]) / head_lengths[i]
                if dist > threshold:
                    true_xs.append(norm_targets[j, 0])
                    true_ys.append(norm_targets[j, 1])
                    pred_xs.append(norm_preds[j, 0])
                    pred_ys.append(norm_preds[j, 1])

    true_xs = np.array(true_xs)
    true_ys = np.array(true_ys)
    pred_xs = np.array(pred_xs)
    pred_ys = np.array(pred_ys)

    xstats = binned_statistic_2d(true_xs, true_ys, pred_xs - true_xs, statistic='mean', bins=8,
                                       range=[[-1, 1], [-1, 1]])
    ystats = binned_statistic_2d(true_xs, true_ys, pred_ys - true_ys, statistic='mean', bins=8,
                                       range=[[-1, 1], [-1, 1]])

    vector_field = np.stack((np.array(xstats.statistic).transpose(), np.array(ystats.statistic).transpose()), axis=-1)

    counts = np.array(binned_statistic_2d(true_xs, true_ys, None, statistic='count', bins=8,
                                       range=[[-1, 1], [-1, 1]]).statistic).transpose()
    totals = np.array(binned_statistic_2d(txs, tys, None, statistic='count', bins=8,
                                       range=[[-1, 1], [-1, 1]]).statistic).transpose()
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.nan_to_num(counts / totals)

    norm = matplotlib.colors.Normalize(vmin=np.min(C), vmax=np.max(C))
    cmap = cubehelix_palette(8, start=0.0, rot=0.85, as_cmap=True)
    smap = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    # This quiver plot shows the average error offset for misclassified joints, binned by
    # grid cell. So the arrows show where the predictions tend to lie relative to the ground
    # truth location. The colour of the arrows indicates the % of misclassified joints for that
    # cell.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, aspect='equal')
    indices = np.unravel_index(np.argsort(C.flatten()), C.shape)
    X = convolve(xstats.x_edge, np.array([0.5, 0.5]), mode='valid')
    Y = convolve(xstats.y_edge, np.array([0.5, 0.5]), mode='valid')
    X, Y = np.meshgrid(X, Y)
    X = X[indices]
    Y = Y[indices]
    U = vector_field[..., 0][indices]
    V = vector_field[..., 1][indices]
    C = C[indices]
    ax.quiver(X, Y, U, V, C, cmap=cmap, angles='xy', scale_units='xy', scale=1, zorder=2)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(1.0, -1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Visualisation of mispredictions')
    ax.set_xticks(xstats.x_edge)
    ax.set_yticks(xstats.y_edge)
    ax.grid(True, zorder=1, linestyle=':')
    cbar = plt.colorbar(smap, format=matplotlib.ticker.PercentFormatter(xmax=1))
    cbar.set_label('Misprediction percentage', rotation=270, labelpad=16)
    plt.show()


if __name__ == '__main__':
    main()
