#!/usr/bin/env python3

"""
This script is for investigating misclassified examples.
"""

import matplotlib
matplotlib.use('TkAgg')

import argparse

import h5py
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.signal import convolve
from scipy.stats import binned_statistic_dd
from seaborn import cubehelix_palette
from torchdata.mpii import MpiiData, transform_keypoints


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='Prediction investigator')
    parser.add_argument('--preds', type=str, metavar='PATH', required=True,
                        help='predictions file')
    parser.add_argument('--subset', type=str, default='val', metavar='S',
                        help='data subset to evaluate on (default="val")')

    args = parser.parse_args()

    return args


def main():
    """Main investigation entrypoint function."""

    args = parse_args()
    preds_file = args.preds
    subset = args.subset

    mpii_data = MpiiData('/datasets/mpii')

    # Load predictions from file
    with h5py.File(preds_file, 'r') as f:
        preds = f['preds'].value.astype(np.float64)

    # Distance threshold for a correct prediction, as a percentage of the head segment length
    threshold = 0.5

    # Ground truth and predicted locations for mispredictions
    mis_true_locs = []
    mis_pred_locs = []
    # All ground truth locations
    all_true_locs = []

    subset_indices = mpii_data.subset_indices(subset)

    for i, index in enumerate(subset_indices):
        true_keypoints = mpii_data.keypoints[index]

        transform_matrix = mpii_data.get_bb_transform(index)

        norm_preds = transform_keypoints(preds[i], transform_matrix)
        norm_targets = transform_keypoints(true_keypoints, transform_matrix)

        # Cutoff distance (in pixels) for "correct" predictions
        distance_cutoff = threshold * mpii_data.head_lengths[index]

        for j in np.nonzero(mpii_data.keypoint_masks[index])[0]:
            if np.abs(norm_targets[j, 0]) <= 1 and np.abs(norm_targets[j, 1]) <= 1:
                all_true_locs.append([norm_targets[j, 0], norm_targets[j, 1]])
                dist = np.linalg.norm(true_keypoints[j] - preds[i, j], ord=2)
                if dist > distance_cutoff:
                    mis_true_locs.append([norm_targets[j, 0], norm_targets[j, 1]])
                    mis_pred_locs.append([norm_preds[j, 0], norm_preds[j, 1]])

    mis_true_locs = np.asarray(mis_true_locs)
    mis_pred_locs = np.asarray(mis_pred_locs)
    all_true_locs = np.asarray(all_true_locs)

    # Calculate relative positions of predictions wrt ground truths
    mean_x_errs = binned_statistic_dd(mis_true_locs, mis_pred_locs[:, 0] - mis_true_locs[:, 0],
                                 statistic='mean', bins=8, range=[[-1, 1], [-1, 1]])
    mean_y_errs = binned_statistic_dd(mis_true_locs, mis_pred_locs[:, 1] - mis_true_locs[:, 1],
                                 statistic='mean', bins=8, range=[[-1, 1], [-1, 1]])
    vector_field = np.stack((np.asarray(mean_x_errs.statistic).transpose(),
                             np.asarray(mean_y_errs.statistic).transpose()), axis=-1)

    # Calculate misprediction percentages
    counts = np.asarray(binned_statistic_dd(mis_true_locs, None, statistic='count', bins=8,
                                          range=[[-1, 1], [-1, 1]]).statistic).transpose()
    totals = np.asarray(binned_statistic_dd(all_true_locs, None, statistic='count', bins=8,
                                          range=[[-1, 1], [-1, 1]]).statistic).transpose()
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.nan_to_num(counts / totals)

    # Create a colour map for per-cell misprediction percentage
    norm = matplotlib.colors.Normalize(vmin=np.min(C), vmax=np.max(C))
    cmap = cubehelix_palette(8, start=0.0, rot=0.85, as_cmap=True)
    smap = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    # This quiver plot shows the average error offset for misclassified joints, binned by
    # grid cell. So the arrows show where the predictions tend to lie relative to the ground
    # truth location. The colour of the arrows indicates the % of misclassified joints for that
    # cell.
    # Calculate the locations of grid cell centres
    X = convolve(mean_x_errs.bin_edges[0], np.asarray([0.5, 0.5]), mode='valid')
    Y = convolve(mean_x_errs.bin_edges[1], np.asarray([0.5, 0.5]), mode='valid')
    X, Y = np.meshgrid(X, Y)
    # Sort everything according to misprediction percentage in order to get nice z-ordering
    indices = np.unravel_index(np.argsort(C.flatten()), C.shape)
    X = X[indices]
    Y = Y[indices]
    U = vector_field[..., 0][indices]
    V = vector_field[..., 1][indices]
    C = C[indices]
    # Create and display the plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, aspect='equal')
    ax.quiver(X, Y, U, V, C, cmap=cmap, angles='xy', scale_units='xy', scale=1, zorder=2)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(1.0, -1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Visualisation of mispredictions')
    ax.set_xticks(mean_x_errs.bin_edges[0])
    ax.set_yticks(mean_x_errs.bin_edges[1])
    ax.grid(True, zorder=1, linestyle=':')
    cbar = plt.colorbar(smap, format=matplotlib.ticker.PercentFormatter(xmax=1))
    cbar.set_label('Misprediction percentage', rotation=270, labelpad=16)
    plt.show()


if __name__ == '__main__':
    main()
