#!/usr/bin/env python3
#
# This script will generate predictions from a trained model.
# It can also calculate PCKh accuracies from predictions for the training and
# validation subsets.
#
# It is expected that the full dataset is available in
# `/data/dlds/mpii-human-pose/`, which should be installed
# using [DLDS](https://github.com/anibali/dlds).
#
# Furthermore, the prediction visualisation functionality in this script
# requires access to the original, unprocessed JPEGs from the official MPII
# dataset release in `/data/dlds/cache/mpii-human-pose/images/`.
# Using `--visualize` is optional.

import torch
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import h5py

import os, sys, inspect
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(cur_dir))

from dsnt.model import build_mpii_pose_model
from dsnt.data import MPIIDataset
from dsnt.eval import PCKhEvaluator

parser = argparse.ArgumentParser(description='DSNT human pose model trainer')
parser.add_argument('--model', type=str, metavar='PATH',
  help='model state file')
parser.add_argument('--preds', type=str, metavar='PATH',
  help='predictions file (will be written to if model is specified)')
parser.add_argument('--subset', type=str, default='val', metavar='S',
  help='data subset to evaluate on (default="val")')
parser.add_argument('--visualize', action='store_true', default=False,
  help='visualize the results')
args = parser.parse_args()

model_file = args.model
preds_file = args.preds
subset = args.subset
visualize = args.visualize

batch_size = 32

# Ground truth
actual_file = '/data/dlds/mpii-human-pose/annot-' + subset + '.h5'
with h5py.File(actual_file, 'r') as f:
  actual = torch.from_numpy(f['part'][:])
  head_lengths = torch.from_numpy(f['normalize'][:])
  imgnames = [name.decode('utf-8') for name in f['imgname'][:]]
joint_mask = actual.select(2, 0).gt(1).mul(actual.select(2, 1).gt(1))

if model_file:
  # Generate predictions with the model

  def progress(count, total):
    bar_len = 60
    filled_len = round(bar_len * count / total)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[{}] {:0.1f}%\r'.format(bar, 100 * count / total))
    sys.stdout.flush()

  model_state = torch.load(model_file)

  model = build_mpii_pose_model(**model_state['model_desc'])
  model.load_state_dict(model_state['state_dict'])
  model.cuda()
  model.eval()

  dataset = MPIIDataset('/data/dlds/mpii-human-pose', subset, use_aug=False)
  loader = DataLoader(dataset, batch_size, num_workers=4, pin_memory=True)
  preds = torch.DoubleTensor(len(dataset), 16, 2)

  completed = 0
  for i, batch in enumerate(loader):
    in_var = Variable(batch['input'].cuda(), requires_grad=False)

    out_var = model(in_var)

    norm_preds = torch.DoubleTensor(out_var.data.size())
    norm_preds.copy_(out_var.data)
    pos = i * batch_size
    orig_preds = preds[pos:pos+norm_preds.size(0)]
    torch.bmm(norm_preds, batch['transform_m'], out=orig_preds)
    orig_preds.add_(batch['transform_b'].expand_as(orig_preds))

    completed += in_var.data.size(0)
    progress(completed, len(dataset))

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
  raise 'at least one of "--preds" and "--model" must be present'

# Calculate PCKh accuracies
evaluator = PCKhEvaluator()
evaluator.add(preds, actual, joint_mask, head_lengths)

# Print PCKh accuracies
for meter_name in sorted(evaluator.meters.keys()):
  meter = evaluator.meters[meter_name]
  mean, std = meter.value()
  print(meter_name, mean)

# Visualise predictions
if visualize:
  import tkinter as tk
  from PIL import ImageTk, Image, ImageDraw
  from dsnt.util import draw_skeleton

  # Directory containing the original MPII human pose image JPEGs
  images_dir = '/data/dlds/cache/mpii-human-pose/images'

  root = tk.Tk()
  img = ImageTk.PhotoImage(Image.open(os.path.join(images_dir, imgnames[0])))
  panel = tk.Label(root)
  panel.pack(side='bottom', fill='both', expand='yes')

  display_opts = {
    'index': 0,
    'ground_truth': True,
  }

  def update_display():
    index = display_opts['index']
    truth = display_opts['ground_truth']
    img = Image.open(os.path.join(images_dir, imgnames[index]))
    if truth:
      draw_skeleton(img, actual[index], joint_mask[index])
    else:
      draw_skeleton(img, preds[index], joint_mask[index])
    draw = ImageDraw.Draw(img)
    draw.text([0, 0], 'Sample {:05d} [{} joints]'.format(
      index, 'actual' if truth else 'predicted'))
    tkimg = ImageTk.PhotoImage(img)
    panel.configure(image=tkimg)
    panel.image = tkimg

  update_display()

  def key(event):
    prev_index = display_opts['index']

    if event.keysym == 'Escape':
      root.destroy()
      return
    if event.keysym == 'Right':
      display_opts['index'] += 1
    if event.keysym == 'Left':
      display_opts['index'] -= 1
    if event.keysym == 'space':
      display_opts['ground_truth'] = not display_opts['ground_truth']

    display_opts['index'] = display_opts['index'] % len(imgnames)
    update_display()

  root.bind('<Key>', key)

  root.mainloop()
