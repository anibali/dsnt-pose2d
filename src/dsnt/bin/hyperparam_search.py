#!/usr/bin/env python3

"""Search for good training hyperparameters.

This code runs the LR range test proposed in "Cyclical Learning Rates for Training Neural Networks"
by Leslie N. Smith.
"""

import argparse
import json

import numpy as np
import pyshowoff
import tele
import torch
from tele.meter import ValueMeter, ListMeter
from tele.showoff.views import Cell, View, Inspect
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from dsnt.data import MPIIDataset
from dsnt.model import build_mpii_pose_model
from dsnt.util import seed_random_number_generators


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model trainer')
    parser.add_argument('--showoff', type=str, default='showoff:3000', metavar='HOST:PORT',
                        help='network location of the Showoff server (default="showoff:3000")')
    # LR finder parameters
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size (default=32)')
    parser.add_argument('--lr-min', type=float, default=1e-1,
                        help='minimum learning rate')
    parser.add_argument('--lr-max', type=float, default=1e2,
                        help='maximum learning rate')
    parser.add_argument('--max-iters', type=int, default=1000,
                        help='number of training iteration')
    parser.add_argument('--ema-beta', type=float, default=0.99,
                        help='beta value for the exponential moving average')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # Model
    parser.add_argument('--base-model', type=str, default='resnet34', metavar='BM',
                        help='base model type (default="resnet34")')
    parser.add_argument('--dilate', type=int, default=0, metavar='N',
                        help='number of ResNet layer groups to dilate (default=0)')
    parser.add_argument('--truncate', type=int, default=0, metavar='N',
                        help='number of ResNet layer groups to cut off (default=0)')
    parser.add_argument('--output-strat', type=str, default='dsnt', metavar='S',
                        choices=['dsnt', 'gauss', 'fc'],
                        help='strategy for outputting coordinates (default="dsnt")')
    parser.add_argument('--preact', type=str, default='softmax', metavar='S',
                        choices=['softmax', 'thresholded_softmax', 'abs', 'relu', 'sigmoid'],
                        help='heatmap preactivation function (default="softmax")')
    parser.add_argument('--reg', type=str, default='none',
                        choices=['none', 'var', 'js', 'kl', 'mse'],
                        help='set the regularizer (default="none")')
    parser.add_argument('--reg-coeff', type=float, default=1.0,
                        help='coefficient for controlling regularization strength')
    parser.add_argument('--hm-sigma', type=float, default=1.0,
                        help='target standard deviation for heatmap, in pixels')
    # RNG
    parser.add_argument('--seed', type=int, metavar='N',
                        help='seed for random number generators')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 999999)

    return args


def make_data_sampler(examples_per_epoch, dataset_length):
    if examples_per_epoch is None:
        examples_per_epoch = dataset_length

    # Sample with replacement only if we have to
    replacement = examples_per_epoch > dataset_length

    return WeightedRandomSampler(
        torch.ones(dataset_length).double(),
        examples_per_epoch,
        replacement=replacement
    )


class _XYGraphCell(Cell):
    def __init__(self, meter_names, frame):
        super().__init__(meter_names, frame)
        self.xs = []
        self.ys = []

    def render(self, step_num, meters):
        series_names = [self.meter_names[0]]
        meter = meters[0]
        assert isinstance(meter, ValueMeter)
        x, y = meter.value()
        self.xs.append(x)
        self.ys.append(y)
        self.frame.line_graph(self.xs, [self.ys], series_names=series_names)


class XYGraph(View):
    def build(self, frame):
        return _XYGraphCell(self.meter_names, frame)


def main():
    args = parse_args()

    seed_random_number_generators(args.seed)

    model_desc = {
        'base': args.base_model,
        'dilate': args.dilate,
        'truncate': args.truncate,
        'output_strat': args.output_strat,
        'preact': args.preact,
        'reg': args.reg,
        'reg_coeff': args.reg_coeff,
        'hm_sigma': args.hm_sigma,
    }
    model = build_mpii_pose_model(**model_desc)
    model.cuda()

    train_data = MPIIDataset('/datasets/mpii', 'train', use_aug=True,
                             image_specs=model.image_specs)
    sampler = make_data_sampler(args.max_iters * args.batch_size, len(train_data))
    train_loader = DataLoader(train_data, args.batch_size, num_workers=4, drop_last=True,
                              sampler=sampler)
    data_iter = iter(train_loader)

    print(json.dumps(model_desc, sort_keys=True, indent=2))

    def do_training_iteration(optimiser):
        batch = next(data_iter)

        in_var = Variable(batch['input'].cuda(), requires_grad=False)
        target_var = Variable(batch['part_coords'].cuda(), requires_grad=False)
        mask_var = Variable(batch['part_mask'].type(torch.cuda.FloatTensor), requires_grad=False)

        # Calculate predictions and loss
        out_var = model(in_var)
        loss = model.forward_loss(out_var, target_var, mask_var)

        # Calculate gradients
        optimiser.zero_grad()
        loss.backward()

        # Update parameters
        optimiser.step()

        return loss.data[0]

    optimiser = SGD(model.parameters(), lr=1, weight_decay=args.weight_decay,
                    momentum=args.momentum)

    tel = tele.Telemetry({
        'cli_args': ValueMeter(skip_reset=True),
        'loss_lr': ValueMeter(),
    })

    tel['cli_args'].set_value(vars(args))

    if args.showoff:
        client = pyshowoff.Client('http://' + args.showoff)
        notebook = client.add_notebook(
            'Hyperparameter search ({}-d{}-t{}, {}, reg={})'.format(
                args.base_model, args.dilate, args.truncate, args.output_strat, args.reg)
        ).result()

        tel.sink(tele.showoff.Conf(notebook), [
            Inspect(['cli_args'], 'CLI arguments', flatten=True),
            XYGraph(['loss_lr'], 'Loss vs learning rate graph'),
        ])

    lrs = np.geomspace(args.lr_min, args.lr_max, args.max_iters)
    avg_loss = 0
    min_loss = np.inf
    for i, lr in enumerate(tqdm(lrs, ascii=True)):
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
        loss = do_training_iteration(optimiser)
        avg_loss = args.ema_beta * avg_loss + (1 - args.ema_beta) * loss
        smoothed_loss = avg_loss / (1 - args.ema_beta ** (i + 1))
        if min_loss > 0 and smoothed_loss > 4 * min_loss:
            break
        min_loss = min(smoothed_loss, min_loss)

        tel['loss_lr'].set_value((lr, smoothed_loss))

        tel.step()


if __name__ == '__main__':
    main()
