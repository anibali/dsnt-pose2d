#!/usr/bin/env python3

'''
This script will train a model on the MPII Human Pose dataset.

It is expected that the full dataset is available in
`/data/dlds/mpii-human-pose/`, which should be installed
using [DLDS](https://github.com/anibali/dlds).
'''

import os
import sys
import inspect
import argparse
import datetime

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torchnet.meter

cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(cur_dir))

from dsnt.nn import EuclideanLoss
from dsnt.data import MPIIDataset
from dsnt.eval import PCKhEvaluator
from dsnt.model import build_mpii_pose_model
from dsnt.visualize import make_dot
from dsnt.util import draw_skeleton
import tele, tele.meter

####
# Options
####

parser = argparse.ArgumentParser(description='DSNT human pose model trainer')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
    help='number of epochs to train (default=200)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
    help='input batch size (default=32)')
parser.add_argument('--showoff', type=str, default='showoff:3000', metavar='HOST:PORT',
    help='network location of the Showoff server (default="showoff:3000")')
parser.add_argument('--no-aug', action='store_true', default=False,
    help='disable training data augmentation')
parser.add_argument('--out-dir', type=str, default='out', metavar='PATH',
    help='path to output directory (default="out")')
parser.add_argument('--base-model', type=str, default='resnet34', metavar='BM',
    help='base model type (default="resnet34")')
parser.add_argument('--truncate', type=int, default=0, metavar='N',
    help='number of ResNet layer groups to cut off (default=0)')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
    help='initial learning rate for SGD (default=0.2)')
parser.add_argument('--schedule-step', type=int, default=50, metavar='N',
    help='number of epochs per LR drop (default=50)')
parser.add_argument('--schedule-gamma', type=float, default=0.5, metavar='G',
    help='factor to multiply the LR by at each drop (default=0.5)')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
showoff_netloc = args.showoff
use_train_aug = not args.no_aug
out_dir = args.out_dir
base_model = args.base_model
truncate = args.truncate
initial_lr = args.lr
schedule_step = args.schedule_step
schedule_gamma = args.schedule_gamma

experiment_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
exp_out_dir = os.path.join(out_dir, experiment_id) if out_dir else None

####
# Model and criterion
####

model_desc = {
    'base': base_model,
    'truncate': truncate,
}
model = build_mpii_pose_model(**model_desc)
model.cuda()

criterion = EuclideanLoss()
criterion.cuda()

####
# Data
####

train_data = MPIIDataset('/data/dlds/mpii-human-pose', 'train', use_aug=use_train_aug)
val_data = MPIIDataset('/data/dlds/mpii-human-pose', 'val', use_aug=False)
train_loader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)

####
# Metrics and visualisation
####

train_eval = PCKhEvaluator()
val_eval = PCKhEvaluator()

def eval_metrics_for_batch(evaluator, batch, norm_out):
    norm_out = norm_out.type(torch.DoubleTensor)

    # Coords in orginal MPII dataset space
    orig_out = torch.bmm(norm_out, batch['transform_m']).add_(
        batch['transform_b'].expand_as(norm_out))

    norm_target = batch['part_coords'].double()
    orig_target = torch.bmm(norm_target, batch['transform_m']).add_(
        batch['transform_b'].expand_as(norm_target))

    head_lengths = batch['normalize'].double()

    evaluator.add(orig_out, orig_target, batch['part_mask'], head_lengths)

tel = tele.Telemetry({
    'experiment_id': tele.meter.ValueMeter(skip_reset=True),
    'epoch': tele.meter.ValueMeter(),
    'train_loss': torchnet.meter.AverageValueMeter(),
    'val_loss': torchnet.meter.AverageValueMeter(),
    'epoch_time': torchnet.meter.TimeMeter(unit=False),
    'val_sample': tele.meter.ValueMeter(),
    'train_sample': tele.meter.ValueMeter(),
    'args': tele.meter.ValueMeter(skip_reset=True),
    'train_pckh_all': train_eval.meters['all'],
    'val_pckh_all': val_eval.meters['all'],
    'val_preds': tele.meter.ValueMeter(),
    'model_graph': tele.meter.ValueMeter(skip_reset=True),
})

# Console output
import tele.console
import tele.console.views as views
meters_to_print = [
    'train_loss', 'val_loss', 'train_pckh_all', 'val_pckh_all', 'epoch_time'
]
tel.sink(tele.console.Conf(), [views.KeyValue([mn]) for mn in meters_to_print])

# Folder output
if exp_out_dir:
    import tele.folder
    import tele.folder.views as views

    tel.sink(tele.folder.Conf(exp_out_dir), [
        views.GrowingJSON(['epoch', 'train_loss', 'val_loss', 'epoch_time',
            'train_pckh_all', 'val_pckh_all'], 'saved_metrics.json'),
        views.HDF5(['val_preds'], 'val_preds_{:04d}.h5', {'val_preds': 'preds'}),
    ])

# Showoff output
progress_frame = None
if showoff_netloc:
    import pyshowoff
    import tele.showoff
    import tele.showoff.views as views

    client = pyshowoff.Client(showoff_netloc)
    notebook = client.new_notebook('Human pose')

    tel.sink(tele.showoff.Conf(notebook), [
        views.LineGraph(['train_loss', 'val_loss'], 'Loss'),
        views.LineGraph(['train_pckh_all', 'val_pckh_all'], 'PCKh all'),
        views.Inspect(['experiment_id', 'epoch', 'train_loss', 'val_loss',
            'train_pckh_all', 'val_pckh_all'], 'Inspect'),
        views.LineGraph(['epoch_time'], 'Time'),
        views.Inspect(['args'], 'Command-line arguments', flatten=True),
        views.Images(['train_sample'], 'Training samples', images_per_row=2),
        views.Images(['val_sample'], 'Validation samples', images_per_row=2),
        views.Graphviz(['model_graph'], 'Model graph'),
    ])

    progress_frame = notebook.new_frame('Progress',
        bounds={'x': 0, 'y': 924, 'width': 1920, 'height': 64})

# Set constant values
tel['experiment_id'].set_value(experiment_id)
tel['args'].set_value(vars(args))

# Generate a Graphviz graph to visualise the model
dummy_data = torch.cuda.FloatTensor(1, 3, 224, 224).uniform_(0, 1)
out_var = model(Variable(dummy_data, requires_grad=False))
tel['model_graph'].set_value(make_dot(out_var, dict(model.named_parameters())))
dummy_data = None

# Initialize optimiser and learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
scheduler = StepLR(optimizer, schedule_step, schedule_gamma)

# `vis` will hold a few samples for visualisation
vis = {}

####
# Training
####

def train(epoch):
    '''Do a full pass over the training set, updating model parameters.'''

    model.train()
    scheduler.step(epoch)
    samples_processed = 0

    for i, batch in enumerate(train_loader):
        in_var = Variable(batch['input'].cuda(), requires_grad=False)
        target_var = Variable(batch['part_coords'].cuda(), requires_grad=False)
        mask_var = Variable(batch['part_mask'].type(torch.cuda.FloatTensor), requires_grad=False)

        optimizer.zero_grad()

        out_var = model(in_var)
        loss = criterion(out_var, target_var, mask_var)
        tel['train_loss'].add(loss.data[0])
        eval_metrics_for_batch(train_eval, batch, out_var.data)

        loss.backward()
        optimizer.step()
        samples_processed += batch['input'].size(0)

        if i == 0:
            vis['train_images'] = batch['input']
            vis['train_preds'] = out_var.data.cpu()
            vis['train_masks'] = batch['part_mask']
            vis['train_coords'] = batch['part_coords']

        if progress_frame is not None:
            progress_frame.progress(epoch * len(train_data) + samples_processed, epochs * len(train_data))

def validate(epoch):
    '''Do a full pass over the validation set, evaluating model performance.'''

    model.eval()
    val_preds = torch.DoubleTensor(len(val_data), 16, 2)

    for i, batch in enumerate(val_loader):
        in_var = Variable(batch['input'].cuda(), requires_grad=False)
        target_var = Variable(batch['part_coords'].cuda(), requires_grad=False)
        mask_var = Variable(batch['part_mask'].type(torch.cuda.FloatTensor), requires_grad=False)

        out_var = model(in_var)
        loss = criterion(out_var, target_var, mask_var)
        tel['val_loss'].add(loss.data[0])
        eval_metrics_for_batch(val_eval, batch, out_var.data)

        preds = torch.DoubleTensor(out_var.data.size())
        preds.copy_(out_var.data)
        pos = i * batch_size
        orig_preds = val_preds[pos:pos+preds.size(0)]
        torch.bmm(preds, batch['transform_m'], out=orig_preds)
        orig_preds.add_(batch['transform_b'].expand_as(preds))

        if i == 0:
            vis['val_images'] = batch['input']
            vis['val_preds'] = preds
            vis['val_masks'] = batch['part_mask']
            vis['val_coords'] = batch['part_coords']

    tel['val_preds'].set_value(val_preds.numpy())

for epoch in range(epochs):
    tel['epoch'].set_value(epoch)
    tel['epoch_time'].reset()

    train(epoch)
    validate(epoch)

    train_sample = []
    for i in range(min(16, vis['train_images'].size(0))):
        img = transforms.ToPILImage()(vis['train_images'][i])
        coords = (vis['train_preds'][i].cpu() + 1) * (224 / 2)
        draw_skeleton(img, coords, vis['train_masks'][i])
        train_sample.append(img)
    tel['train_sample'].set_value(train_sample)

    val_sample = []
    for i in range(min(16, vis['val_images'].size(0))):
        img = transforms.ToPILImage()(vis['val_images'][i])
        coords = (vis['val_preds'][i].cpu() + 1) * (224 / 2)
        draw_skeleton(img, coords, vis['val_masks'][i])
        val_sample.append(img)
    tel['val_sample'].set_value(val_sample)

    if exp_out_dir:
        state = {
            'state_dict': model.state_dict(),
            'model_desc': model_desc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(state, os.path.join(exp_out_dir, 'model.pth'))

    tel.step()
    train_eval.reset()
    val_eval.reset()
