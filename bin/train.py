#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchvision.models import resnet34
import torchvision.transforms as transforms
from PIL.ImageDraw import Draw
from torch.utils import model_zoo
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torchnet.meter
import gc
import argparse
import datetime

import os, sys, inspect
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(cur_dir)) 

from dsnt.nn import DSNT, EuclideanLoss
from dsnt.data import MPIIDataset
from dsnt.eval import PCKhEvaluator
import tele, tele.meter, tele.output.console, tele.output.folder

parser = argparse.ArgumentParser(description='DSNT human pose model trainer')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
  help='number of epochs to train (default=100)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
  help='input batch size (default=32)')
parser.add_argument('--showoff', type=str, default='showoff:3000', metavar='HOST:PORT',
  help='network location of the Showoff server (default="showoff:3000")')
parser.add_argument('--no-aug', action='store_true', default=False,
  help='disable training data augmentation')
parser.add_argument('--out-dir', type=str, default='out', metavar='PATH',
  help='path to output directory (default="out")')
parser.add_argument('--truncate', type=int, default=0, metavar='N',
  help='number of ResNet layer groups to cut off (default=0)')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
showoff_netloc = args.showoff
use_train_aug = not args.no_aug
out_dir = args.out_dir
truncate = args.truncate

initial_lr = 0.1
experiment_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')

ResNet34_URL = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'

class ResNetLocalizer(nn.Module):
  def __init__(self, resnet, n_chans=1, truncate=0):
    super(ResNetLocalizer, self).__init__()
    self.n_chans = n_chans
    fcn_modules = [
      resnet.conv1,
      resnet.bn1,
      resnet.relu,
      resnet.maxpool,
    ]
    resnet_groups = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    fcn_modules.extend(resnet_groups[:len(resnet_groups)-truncate])
    self.fcn = nn.Sequential(*fcn_modules)
    feats = fcn_modules[-1][0].conv1.out_channels
    self.hm_conv = nn.Conv2d(feats, self.n_chans, kernel_size=1, bias=False)
    self.hm_preact = nn.Softmax()
    self.hm_dsnt = DSNT()
    self.out_size = 7 * (2 ** truncate)
  
  def forward(self, x):
    x = self.fcn(x)
    x = self.hm_conv(x)
    x = x.view(-1, self.out_size*self.out_size)
    x = self.hm_preact(x)
    x = x.view(-1, self.n_chans, self.out_size, self.out_size)
    x = self.hm_dsnt(x)
    return x

resnet34_model = resnet34()
resnet34_model.load_state_dict(model_zoo.load_url(ResNet34_URL, './models'))
model = ResNetLocalizer(resnet34_model, n_chans=16, truncate=truncate)
model.cuda()

criterion = EuclideanLoss()
criterion.cuda()

train_data = MPIIDataset('/data/dlds/mpii-human-pose', 'train', use_aug=use_train_aug)
val_data = MPIIDataset('/data/dlds/mpii-human-pose', 'val', use_aug=False)
train_loader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)

train_eval = PCKhEvaluator()
val_eval = PCKhEvaluator()

def eval_metrics_for_batch(evaluator, batch, norm_out):
  norm_out = norm_out.type(torch.DoubleTensor)

  # Coords in orginal MPII dataset space
  orig_out = torch.bmm(norm_out, batch['transform_m'].double()).add_(
    batch['transform_b'].double().expand_as(norm_out))

  norm_target = batch['part_coords'].double()
  orig_target = torch.bmm(norm_target, batch['transform_m'].double()).add_(
    batch['transform_b'].double().expand_as(norm_target))

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
})

console_output = tele.output.console.ConsoleOutput()
meters_to_print = [
  'train_loss', 'val_loss', 'train_pckh_all', 'val_pckh_all', 'epoch_time'
]
console_output.set_cells([([mn], tele.output.console.TextCell()) for mn in meters_to_print])
console_output.set_auto_default_cell(False)
tel.add_output(console_output)

if out_dir:
  folder_output = tele.output.folder.FolderOutput(os.path.join('out', experiment_id))
  folder_output.set_cells([(
    ['epoch', 'train_loss', 'val_loss', 'epoch_time', 'train_pckh_all', 'val_pckh_all'],
    tele.output.folder.GrowingJSONCell('saved_metrics.json')
  )])
  tel.add_output(folder_output)

if showoff_netloc:
  import pyshowoff, tele.output.showoff

  client = pyshowoff.Client(showoff_netloc)
  notebook = client.new_notebook('Human pose')

  showoff_output = tele.output.showoff.ShowoffOutput(notebook)
  showoff_output.set_cells([
    (['train_loss', 'val_loss'],
      tele.output.showoff.LineGraphCell('Loss')),
    (['train_pckh_all', 'val_pckh_all'],
      tele.output.showoff.LineGraphCell('PCKh all')),
    (['experiment_id', 'epoch', 'train_loss', 'val_loss', 'train_pckh_all', 'val_pckh_all'],
      tele.output.showoff.InspectValueCell('Inspect')),
    (['epoch_time'],
      tele.output.showoff.LineGraphCell('Time')),
    (['args'],
      tele.output.showoff.InspectValueCell('Command-line arguments', flatten=True)),
    (['train_sample'],
      tele.output.showoff.ImageCell('Training samples', images_per_row=2)),
    (['val_sample'],
      tele.output.showoff.ImageCell('Validation samples', images_per_row=2)),
  ])
  showoff_output.set_auto_default_cell(False)
  tel.add_output(showoff_output)

  progress_frame = notebook.new_frame('Progress',
    bounds={'x': 0, 'y': 924, 'width': 1920, 'height': 64})
else:
  progress_frame = None

def adjust_learning_rate(optimizer, epoch):
  decay_factor = 0.5
  step_width = 25
  lr = initial_lr * (decay_factor ** (epoch // step_width))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)

vis = {}

def train(epoch):
  model.train()
  adjust_learning_rate(optimizer, epoch)
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
  model.eval()

  for i, batch in enumerate(val_loader):
    in_var = Variable(batch['input'].cuda(), requires_grad=False)
    target_var = Variable(batch['part_coords'].cuda(), requires_grad=False)
    mask_var = Variable(batch['part_mask'].type(torch.cuda.FloatTensor), requires_grad=False)

    out_var = model(in_var)
    loss = criterion(out_var, target_var, mask_var)
    tel['val_loss'].add(loss.data[0])
    eval_metrics_for_batch(val_eval, batch, out_var.data)

    if i == 0:
      vis['val_images'] = batch['input']
      vis['val_preds'] = out_var.data.cpu()
      vis['val_masks'] = batch['part_mask']
      vis['val_coords'] = batch['part_coords']

# Joints to connect for visualisation, giving the effect of drawing a
# basic "skeleton" of the pose.
bones = {
  'right_lower_leg':    (0, 1),
  'right_upper_leg':    (1, 2),
  'right_pelvis':       (2, 6),
  'left_lower_leg':     (4, 5),
  'left_upper_leg':     (3, 4),
  'left_pelvis':        (3, 6),
  'center_lower_torso': (6, 7),
  'center_upper_torso': (7, 8),
  'center_head':        (8, 9),
  'right_lower_arm':    (10, 11),
  'right_upper_arm':    (11, 12),
  'right_shoulder':     (12, 8),
  'left_lower_arm':     (14, 15),
  'left_upper_arm':     (13, 14),
  'left_shoulder':      (13, 8),
}

def draw_skeleton_(img, part_coords, joint_mask=None):
  coords = (part_coords.cpu() + 1) * (224 / 2)
  draw = Draw(img)
  for bone_name, (j1, j2) in bones.items():
    if bone_name.startswith('center_'):
      colour = (255, 0, 255)  # Magenta
    elif bone_name.startswith('left_'):
      colour = (0, 0, 255)    # Blue
    elif bone_name.startswith('right_'):
      colour = (255, 0, 0)    # Red
    else:
      colour = (255, 255, 255)
    
    if joint_mask is not None:
      # Change colour to grey if either vertex is not masked in
      if joint_mask[j1] == 0 or joint_mask[j2] == 0:
        colour = (100, 100, 100)

    draw.line([coords[j1, 0], coords[j1, 1], coords[j2, 0], coords[j2, 1]], fill=colour)

tel['experiment_id'].set_value(experiment_id)
tel['args'].set_value(vars(args))

for epoch in range(epochs):
  tel['epoch'].set_value(epoch)
  tel['epoch_time'].reset()

  train(epoch)
  gc.collect()
  validate(epoch)
  gc.collect()

  train_sample = []
  for i in range(min(16, vis['train_images'].size(0))):
    img = transforms.ToPILImage()(vis['train_images'][i])
    draw_skeleton_(img, vis['train_preds'][i], vis['train_masks'][i])
    train_sample.append(img)
  tel['train_sample'].set_value(train_sample)

  val_sample = []
  for i in range(min(16, vis['val_images'].size(0))):
    img = transforms.ToPILImage()(vis['val_images'][i])
    draw_skeleton_(img, vis['val_preds'][i], vis['val_masks'][i])
    val_sample.append(img)
  tel['val_sample'].set_value(val_sample)

  tel.step()
  train_eval.reset()
  val_eval.reset()
