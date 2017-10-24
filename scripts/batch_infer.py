#!/usr/bin/env python3

"""
This script will benchmark a trained model.

The model will be measured for speed and accuracy on the validation set.
"""

import os
import random
import torch
import h5py
import numpy as np
import json
from tele.meter import MedianValueMeter

from dsnt.model import build_mpii_pose_model
from dsnt.data import MPIIDataset
from dsnt.evaluator import PCKhEvaluator
from dsnt.inference import generate_predictions, evaluate_mpii_predictions


def seed_random_number_generators(seed):
    """Seed all random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """Main benchmark entrypoint function."""

    in_dir = 'out/by-alias'
    subset = 'val'

    seed_random_number_generators(12345)

    files = {os.path.realpath(os.path.join(in_dir, f)) for f in os.listdir(in_dir)}
    exp_dirs = [d for d in files if os.path.isdir(d)]

    for exp_dir in sorted(exp_dirs):
        print(exp_dir)
        model_file = os.path.join(exp_dir, 'model.pth')
        preds_file = os.path.join(exp_dir, 'infer-{}.h5'.format(subset))
        metrics_file = os.path.join(exp_dir, 'infer-{}-metrics.json'.format(subset))
        if not os.path.isfile(model_file):
            print('cannot find model.pth')
            continue
        if os.path.isfile(preds_file):
            print('predictions found, skipping')
            continue

        model_state = torch.load(model_file)
        model_desc = model_state['model_desc']
        model = build_mpii_pose_model(**model_desc)
        model.load_state_dict(model_state['state_dict'])

        print(model_desc)

        dataset = MPIIDataset('/data/dlds/mpii-human-pose', subset,
                              use_aug=False, image_specs=model.image_specs)

        inference_time_meter = MedianValueMeter()

        preds = generate_predictions(model, dataset, use_flipped=False,
                                     time_meter=inference_time_meter, batch_size=1)

        # Save predictions to file
        with h5py.File(preds_file, 'w') as f:
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

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, sort_keys=True, indent=2, separators=(',', ': '))


if __name__ == '__main__':
    main()
