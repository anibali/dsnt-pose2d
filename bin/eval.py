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

from dsnt.model import build_mpii_pose_model, ResNetHumanPoseModel
from dsnt.data import MPIIDataset
from dsnt.evaluator import PCKhEvaluator
import dsnt.gui
from dsnt.util import type_as_index, reverse_tensor


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model evaluator')
    parser.add_argument('--model', type=str, metavar='PATH',
                        help='model state file')
    parser.add_argument('--preds', type=str, metavar='PATH',
                        help='predictions file (will be written to if model is specified)')
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

        use_flipped = True
        if args.disable_flip or not isinstance(model, ResNetHumanPoseModel):
            use_flipped = False

        print('Using flip augmentations? {}'.format(use_flipped))

        dataset = MPIIDataset('/data/dlds/mpii-human-pose', subset,
            use_aug=False, image_specs=model.image_specs)
        loader = DataLoader(dataset, batch_size, num_workers=4, pin_memory=True)
        preds = torch.DoubleTensor(len(dataset), 16, 2).zero_()

        completed = 0
        with progressbar.ProgressBar(max_value=len(dataset)) as bar:
            for i, batch in enumerate(loader):
                batch_size = batch['input'].size(0)

                if use_flipped:
                    # Normal
                    in_var = Variable(batch['input'].cuda(), volatile=True)
                    hm_var = model.forward_part1(in_var)
                    hm1 = Variable(hm_var.data.clone(), volatile=True)

                    # Flipped
                    in_var = Variable(reverse_tensor(batch['input'], -1).cuda(), volatile=True)
                    hm_var = model.forward_part1(in_var)
                    hm2 = reverse_tensor(hm_var, -1)
                    hm2 = hm2.index_select(-3, type_as_index(MPIIDataset.HFLIP_INDICES, hm2))

                    hm = (hm1 + hm2) / 2
                    out_var = model.forward_part2(hm)
                    coords = model.compute_coords(out_var)
                    orig_preds = torch.baddbmm(
                        batch['transform_b'],
                        coords.double(),
                        batch['transform_m'])
                else:
                    in_var = Variable(batch['input'].cuda(), volatile=True)
                    out_var = model(in_var)
                    coords = model.compute_coords(out_var)
                    orig_preds = torch.baddbmm(
                        batch['transform_b'],
                        coords.double(),
                        batch['transform_m'])

                pos = i * batch_size
                preds[pos:(pos + batch_size)] = orig_preds

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
