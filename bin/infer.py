#!/usr/bin/env python3

"""
This script will benchmark a trained model.

The model will be measured for speed and accuracy on the validation set.
"""

import argparse
import random
import time

import torch
import h5py
import numpy as np

from dsnt.meter import MedianValueMeter
from dsnt.model import build_mpii_pose_model
from dsnt.data import MPIIDataset
from dsnt.evaluator import PCKhEvaluator
from dsnt.inference import generate_predictions, evaluate_mpii_predictions


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model evaluator')
    parser.add_argument(
        '--model', type=str, metavar='PATH', required=True,
        help='model state file')
    parser.add_argument(
        '--output', '-o', type=str, metavar='PATH',
        help='output file to write predictions to')
    parser.add_argument(
        '--subset', type=str, default='val', metavar='S',
        help='data subset to evaluate on (default="val")')
    parser.add_argument(
        '--disable-flip', action='store_true', default=False,
        help='disable the use of horizontally flipped images')
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


def main():
    """Main benchmark entrypoint function."""

    args = parse_args()
    seed_random_number_generators(args.seed)

    model_file = args.model
    preds_file = args.output
    subset = args.subset

    model_state = torch.load(model_file)
    model = build_mpii_pose_model(**model_state['model_desc'])
    model.load_state_dict(model_state['state_dict'])

    print(model_state['model_desc'])

    use_flipped = not args.disable_flip

    print('Use flip augmentations: {}'.format(use_flipped))

    dataset = MPIIDataset('/data/dlds/mpii-human-pose', subset,
                          use_aug=False, image_specs=model.image_specs)

    inference_time_meter = MedianValueMeter()

    preds = generate_predictions(model, dataset, use_flipped=use_flipped,
                                 time_meter=inference_time_meter, batch_size=1)

    # Save predictions to file
    if preds_file:
        with h5py.File(preds_file, 'w') as f:
            f.create_dataset('preds', data=preds.float().numpy())

    # PyCharm console output breaks unless we pause here briefly
    time.sleep(0.2)

    # Print inference time per image
    time_mean, time_std = inference_time_meter.value()
    print()
    print('Inference time: {:0.2f}±{:0.2f} ms'.format(time_mean * 1000, time_std * 1000))

    # Calculate and print PCKh accuracy
    evaluator = PCKhEvaluator()
    evaluate_mpii_predictions(preds, subset, evaluator)
    print()
    print('# Accuracy (PCKh)')
    print('all: {:0.6f}'.format(evaluator.meters['all'].value()[0]))
    print('total_mpii: {:0.6f}'.format(evaluator.meters['total_mpii'].value()[0]))
    print('total_anewell: {:0.6f}'.format(evaluator.meters['total_anewell'].value()[0]))


if __name__ == '__main__':
    main()
