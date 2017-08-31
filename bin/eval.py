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

import progressbar
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import h5py
import numpy as np

from dsnt.model import build_mpii_pose_model
from dsnt.data import MPIIDataset
from dsnt.eval import PCKhEvaluator
import dsnt.gui


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model evaluator')
    parser.add_argument('--model', type=str, metavar='PATH',
        help='model state file')
    parser.add_argument('--preds', type=str, metavar='PATH',
        help='predictions file (will be written to if model is specified)')
    parser.add_argument('--subset', type=str, default='val', metavar='S',
        help='data subset to evaluate on (default="val")')
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

    # Ground truth
    annot_file = '/data/dlds/mpii-human-pose/annot-{}.h5'.format(subset)
    with h5py.File(annot_file, 'r') as f:
        actual = torch.from_numpy(f['part'][:])
        head_lengths = torch.from_numpy(f['normalize'][:])
    joint_mask = actual.select(2, 0).gt(1).mul(actual.select(2, 1).gt(1))

    model = None
    if model_file:
        # Generate predictions with the model

        model_state = torch.load(model_file)

        model = build_mpii_pose_model(**model_state['model_desc'])
        model.load_state_dict(model_state['state_dict'])
        model.cuda()
        model.eval()

        dataset = MPIIDataset('/data/dlds/mpii-human-pose', subset,
            use_aug=False, size=model.input_size)
        loader = DataLoader(dataset, batch_size, num_workers=4, pin_memory=True)
        preds = torch.DoubleTensor(len(dataset), 16, 2).zero_()

        completed = 0
        with progressbar.ProgressBar(max_value=len(dataset)) as bar:
            for i, batch in enumerate(loader):
                in_var = Variable(batch['input'].cuda(), requires_grad=False)

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

        # Save predictions to file
        if preds_file:
            with h5py.File(preds_file, 'w') as f:
                f.create_dataset('preds', data=preds.numpy())
    elif preds_file:
        # Load predictions from file
        with h5py.File(preds_file, 'r') as f:
            preds = torch.from_numpy(f['preds'][:])
    else:
        # We need to get predictions from somewhere!
        raise Exception('at least one of "--preds" and "--model" must be present')

    # Calculate PCKh accuracies
    evaluator = PCKhEvaluator()
    evaluator.add(preds, actual, joint_mask, head_lengths)

    # Print PCKh accuracies
    for meter_name in sorted(evaluator.meters.keys()):
        meter = evaluator.meters[meter_name]
        mean, _ = meter.value()
        print(meter_name, mean)

    # Visualise predictions
    if visualize:
        dsnt.gui.run_gui(preds, annot_file, model)

if __name__ == '__main__':
    main()
