#!/usr/bin/env python3

"""
This script will generate predictions and evaluation metrics for
multiple models.
"""

import argparse
import json
from pathlib import Path

import h5py
import torch
from tele.meter import MedianValueMeter

from dsnt.data import MPIIDataset
from dsnt.evaluator import PCKhEvaluator
from dsnt.inference import generate_predictions, evaluate_mpii_predictions
from dsnt.model import build_mpii_pose_model
from dsnt.util import seed_random_number_generators


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model evaluator')
    parser.add_argument(
        '--search-dir', type=str, default='out/', metavar='PATH',
        help='base directory to search for models in (default="out/")')
    parser.add_argument(
        '--subset', type=str, default='val', metavar='S',
        help='data subset to evaluate on (default="val")')

    args = parser.parse_args()

    return args


def main():
    """Main benchmark entrypoint function."""

    args = parse_args()

    in_dir = Path(args.search_dir)
    subset = args.subset

    seed_random_number_generators(12345)

    exp_dirs = [
        candidate.parent
        for candidate in in_dir.rglob('model.pth')
        if candidate.is_file()
    ]

    for exp_dir in sorted(exp_dirs):
        model_file = exp_dir / 'model.pth'
        preds_file = exp_dir / 'infer-{}.h5'.format(subset)
        metrics_file = exp_dir / 'infer-{}-metrics.json'.format(subset)
        if not model_file.is_file():
            print('cannot find model.pth')
            continue
        if preds_file.is_file():
            print('predictions found, skipping')
            continue

        model_state = torch.load(str(model_file))
        model_desc = model_state['model_desc']
        model = build_mpii_pose_model(**model_desc)
        model.load_state_dict(model_state['state_dict'])

        print(model_desc)

        dataset = MPIIDataset('/datasets/mpii', subset,
                              use_aug=False, image_specs=model.image_specs)

        inference_time_meter = MedianValueMeter()

        preds = generate_predictions(model, dataset, use_flipped=False,
                                     time_meter=inference_time_meter, batch_size=1)

        # Save predictions to file
        with h5py.File(str(preds_file), 'w') as f:
            f.create_dataset('preds', data=preds.float().numpy())

        time_median, time_err = inference_time_meter.value()
        print('Inference time: {:0.2f}±{:0.2f} ms'.format(time_median * 1000, time_err * 1000))

        evaluator = PCKhEvaluator()
        evaluate_mpii_predictions(preds, subset, evaluator)

        metrics = {
            'inference_time_ms': {
                'median': time_median * 1000,
                'error': time_err * 1000, # Median absolute deviation
            },
            'accuracy_pckh': {
                'all': evaluator.meters['all'].value()[0],
                'total_mpii': evaluator.meters['total_mpii'].value()[0],
                'total_anewell': evaluator.meters['total_anewell'].value()[0],
            },
        }

        with metrics_file.open('w') as f:
            json.dump(metrics, f, sort_keys=True, indent=2, separators=(',', ': '))


if __name__ == '__main__':
    main()
