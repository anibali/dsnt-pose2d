#!/usr/bin/env python3

"""
This script will generate predictions from a trained model.
It can also calculate PCKh accuracies from predictions for the training and
validation subsets.

It is expected that the full dataset is available in
`/data/dlds/mpii-human-pose/`, which should be installed
using [DLDS](https://github.com/anibali/dlds).

Furthermore, the prediction visualisation functionality in this script
requires access to the original, unprocessed JPEGs from the official MPII
dataset release in `/data/dlds/cache/mpii-human-pose/images/`.
Using `--visualize` is optional.
"""

import argparse
import random

import torch
import h5py
import numpy as np

from dsnt.model import build_mpii_pose_model
from dsnt.data import MPIIDataset
from dsnt.evaluator import PCKhEvaluator
import dsnt.gui
from dsnt.inference import generate_predictions, evaluate_mpii_predictions


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model evaluator')
    parser.add_argument('--model', type=str, metavar='PATH',
                        help='model state file')
    parser.add_argument('--preds', type=str, metavar='PATH',
                        help='predictions file')
    parser.add_argument('--subset', type=str, default='val', metavar='S',
                        help='data subset to evaluate on (default="val")')
    parser.add_argument('--disable-flip', action='store_true', default=False,
                        help='disable the use of horizontally flipped images')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='visualize the results')
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

    model_file = args.model
    preds_file = args.preds
    subset = args.subset
    visualize = args.visualize

    batch_size = 6

    model = None
    if model_file:
        model_state = torch.load(model_file)
        model = build_mpii_pose_model(**model_state['model_desc'])
        model.load_state_dict(model_state['state_dict'])
        model = model.cuda()
        print(model_state['model_desc'])

    if preds_file:
        # Load predictions from file
        with h5py.File(preds_file, 'r') as f:
            preds = torch.from_numpy(f['preds'][:]).double()
    elif model:
        # Generate predictions with the model
        use_flipped = not args.disable_flip
        print('Use flip augmentations: {}'.format(use_flipped))
        dataset = MPIIDataset(
            '/data/dlds/mpii-human-pose', subset, use_aug=False, image_specs=model.image_specs)
        preds = generate_predictions(
            model, dataset, use_flipped=use_flipped, batch_size=batch_size)
    else:
        # We need to get predictions from somewhere!
        raise Exception('at least one of "--preds" and "--model" must be present')

    # Calculate PCKh accuracies
    evaluator = PCKhEvaluator()
    evaluate_mpii_predictions(preds, subset, evaluator)

    # Print PCKh accuracies
    for meter_name in sorted(evaluator.meters.keys()):
        meter = evaluator.meters[meter_name]
        mean, _ = meter.value()
        print(meter_name, mean)

    # Visualise predictions
    if visualize:
        annot_file = '/data/dlds/mpii-human-pose/annot-{}.h5'.format(subset)
        dsnt.gui.run_gui(preds, annot_file, model)


if __name__ == '__main__':
    main()
