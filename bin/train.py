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

import os, sys, inspect
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(cur_dir)) 

from dsnt.nn import DSNT, EuclideanLoss
from dsnt.data import MPIIDataset
import tele, tele.output.console, tele.meter

parser = argparse.ArgumentParser(description='DSNT human pose model trainer')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
  help='number of epochs to train (default=100)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
  help='input batch size (default=32)')
parser.add_argument('--showoff', type=str, default='showoff:3000', metavar='host:port',
  help='network location of the Showoff server (default="showoff:3000")')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
showoff_netloc = args.showoff

ResNet34_URL = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'

class ResNetLocalizer(nn.Module):
  def __init__(self, resnet, n_chans=1):
    super(ResNetLocalizer, self).__init__()
    self.n_chans = n_chans
    self.fcn = nn.Sequential(
      resnet.conv1,
      resnet.bn1,
      resnet.relu,
      resnet.maxpool,
      resnet.layer1,
      resnet.layer2,
      resnet.layer3,
      resnet.layer4
    )
    self.hm_conv = nn.Conv2d(resnet34_model.fc.in_features, self.n_chans, kernel_size=1, bias=False)
    self.hm_preact = nn.Softmax()
    self.hm_dsnt = DSNT()
  
  def forward(self, x):
    x = self.fcn(x)
    x = self.hm_conv(x)
    x = x.view(-1, 7*7)
    x = self.hm_preact(x)
    x = x.view(-1, self.n_chans, 7, 7)
    x = self.hm_dsnt(x)
    return x

resnet34_model = resnet34()
resnet34_model.load_state_dict(model_zoo.load_url(ResNet34_URL, './models'))
model = ResNetLocalizer(resnet34_model, 16)
model.cuda()

criterion = EuclideanLoss()
criterion.cuda()

train_data = MPIIDataset('/data/dlds/mpii-human-pose', 'train', use_aug=True)
val_data = MPIIDataset('/data/dlds/mpii-human-pose', 'val', use_aug=False)
in_var = None

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

tel = tele.Telemetry({
  'train_loss': torchnet.meter.AverageValueMeter(),
  'val_loss': torchnet.meter.AverageValueMeter(),
  'epoch_time': torchnet.meter.TimeMeter(unit=False),
  'val_sample': tele.meter.ValueMeter(),
  'args': tele.meter.ValueMeter(),
  'epoch': tele.meter.ValueMeter(),
})

console_output = tele.output.console.ConsoleOutput()
meters_to_print = ['train_loss', 'epoch_time']
console_output.set_cells([([mn], tele.output.console.TextCell()) for mn in meters_to_print])
console_output.set_auto_default_cell(False)
tel.add_output(console_output)

if showoff_netloc:
  import pyshowoff, tele.output.showoff

  client = pyshowoff.Client(showoff_netloc)
  notebook = client.new_notebook('Human pose')

  showoff_output = tele.output.showoff.ShowoffOutput(notebook)
  showoff_output.set_cells([
    (['train_loss', 'val_loss'], tele.output.showoff.LineGraphCell('Loss')),
    (['epoch', 'train_loss', 'val_loss'], tele.output.showoff.InspectValueCell('Inspect')),
    (['epoch_time'], tele.output.showoff.LineGraphCell('Time')),
    (['val_sample'], tele.output.showoff.ImageCell('Validation samples', images_per_row=2)),
    (['args'], tele.output.showoff.InspectValueCell('Command-line arguments', flatten=True)),
  ])
  showoff_output.set_auto_default_cell(False)
  tel.add_output(showoff_output)

  progress_frame = notebook.new_frame('Progress',
    bounds={'x': 0, 'y': 924, 'width': 1920, 'height': 64})
else:
  progress_frame = None

def train(epoch):
  model.train()
  loader = DataLoader(train_data, batch_size, num_workers=4)
  samples_processed = 0

  for batch in loader:
    in_var = Variable(batch['input'].cuda(), requires_grad=False)
    target_var = Variable(batch['part_coords'].cuda(), requires_grad=False)
    mask_var = Variable(batch['part_mask'].type(torch.cuda.FloatTensor), requires_grad=False)

    optimizer.zero_grad()
    out_var = model(in_var)
    loss = criterion(out_var, target_var, mask_var)
    tel['train_loss'].add(loss.data[0])
    loss.backward()
    optimizer.step()
    samples_processed += batch['input'].size(0)

    if progress_frame is not None:
      progress_frame.progress(epoch * len(train_data) + samples_processed, epochs * len(train_data))

vis = {}
def validate(epoch):
  model.eval()
  loader = DataLoader(val_data, batch_size, num_workers=4)

  for i, batch in enumerate(loader):
    in_var = Variable(batch['input'].cuda(), requires_grad=False)
    target_var = Variable(batch['part_coords'].cuda(), requires_grad=False)
    mask_var = Variable(batch['part_mask'].type(torch.cuda.FloatTensor), requires_grad=False)

    out_var = model(in_var)
    loss = criterion(out_var, target_var, mask_var)
    tel['val_loss'].add(loss.data[0])

    if i == 0:
      vis['images'] = batch['input']
      vis['preds'] = out_var.data.cpu()
      vis['masks'] = batch['part_mask']

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
      colour = (255, 0, 255)
    elif bone_name.startswith('left_'):
      colour = (0, 0, 255)
    elif bone_name.startswith('right_'):
      colour = (255, 0, 0)
    else:
      colour = (255, 255, 255)
    
    if joint_mask is not None:
      # Change colour to grey if either vertex is not masked in
      if joint_mask[j1] < 0.1 or joint_mask[j2] < 0.1:
        colour = (100, 100, 100)

    draw.line([coords[j1, 0], coords[j1, 1], coords[j2, 0], coords[j2, 1]], fill=colour)

for epoch in range(epochs):
  tel['epoch'].set_value(epoch)
  tel['args'].set_value(vars(args))
  tel['epoch_time'].reset()

  train(epoch)
  gc.collect()
  validate(epoch)
  gc.collect()

  val_sample = []
  for i in range(min(16, vis['images'].size(0))):
    img = transforms.ToPILImage()(vis['images'][i])
    draw_skeleton_(img, vis['preds'][i], vis['masks'][i])
    val_sample.append(img)
  tel['val_sample'].set_value(val_sample)

  tel.step()
