#!/usr/bin/env python3

'''
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
'''

import os
import sys
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

def parse_args():
    'Parse command-line arguments.'

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
    'Seed all random number generators.'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    'Main evaluation entrypoint function.'

    args = parse_args()
    seed_random_number_generators(args.seed)

    model_file = args.model
    preds_file = args.preds
    subset = args.subset
    visualize = args.visualize

    batch_size = 6

    # Ground truth
    actual_file = '/data/dlds/mpii-human-pose/annot-' + subset + '.h5'
    with h5py.File(actual_file, 'r') as f:
        actual = torch.from_numpy(f['part'][:])
        head_lengths = torch.from_numpy(f['normalize'][:])
        imgnames = [name.decode('utf-8') for name in f['imgname'][:]]
        centers = torch.from_numpy(f['center'][:])
        scales = torch.from_numpy(f['scale'][:])
    joint_mask = actual.select(2, 0).gt(1).mul(actual.select(2, 1).gt(1))

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

    n_joints = preds.size(1)
    scores = []

    # Print PCKh accuracies
    for meter_name in sorted(evaluator.meters.keys()):
        meter = evaluator.meters[meter_name]
        mean, _ = meter.value()
        print(meter_name, mean)

    # Compute interesting orderings
    for b in range(preds.size(0)):
        dists = []
        for j in range(n_joints):
            if joint_mask[b, j] == 1:
                dist = torch.dist(actual[b, j], preds[b, j]) / head_lengths[b]
                dists.append(dist)
        if len(dists) > 0:
            scores.append(sum(dists) / len(dists))
        else:
            scores.append(0)
    accuracy_ordering = list(np.argsort(np.array(scores))[::-1])
    identity_ordering = list(range(preds.size(0)))

    # Visualise predictions
    if visualize:
        import tkinter as tk
        import tkinter.font
        import tkinter.filedialog
        from PIL import ImageTk, Image, ImageDraw
        from dsnt.util import draw_skeleton

        # Directory containing the original MPII human pose image JPEGs
        images_dir = '/data/dlds/cache/mpii-human-pose/images'

        class PoseResultsFrame(tk.Frame):
            SKELETON_NONE = 'None'
            SKELETON_TRUTH = 'Ground truth'
            SKELETON_PREDICTION = 'Prediction'

            def __init__(self):
                super().__init__()

                self.ordering = accuracy_ordering

                self.init_gui()
                self.savable_image = None

            @property
            def cur_sample(self):
                return int(self.var_cur_sample.get())

            @cur_sample.setter
            def cur_sample(self, value):
                self.var_cur_sample.set(str(value))

            @property
            def crop_as_input(self):
                return self.var_crop_as_input.get() == 1

            @crop_as_input.setter
            def crop_as_input(self, value):
                self.var_crop_as_input.set(1 if value else 0)

            def update_image(self):
                index = self.ordering[self.cur_sample]
                self.var_index.set('Index: {:04d}'.format(index))
                img = Image.open(os.path.join(images_dir, imgnames[index]))

                if self.var_skeleton.get() == self.SKELETON_TRUTH:
                    draw_skeleton(img, actual[index], joint_mask[index])
                elif self.var_skeleton.get() == self.SKELETON_PREDICTION:
                    draw_skeleton(img, preds[index], joint_mask[index])

                # Calculate crop used for input
                scale = scales[index]
                center = centers[index].clone()
                center[1] += 15 * scale
                scale *= 1.25
                size = scale * 200

                if self.crop_as_input:
                    img = img.crop([center[0] - size/2, center[1] - size/2,
                        center[0] + size/2, center[1] + size/2])

                self.savable_image = img.copy()

                width = self.image_panel.winfo_width()
                height = self.image_panel.winfo_height() - 2
                img.thumbnail((width, height), Image.ANTIALIAS)
                tkimg = ImageTk.PhotoImage(img)
                self.image_panel.configure(image=tkimg)
                self.image_panel.image = tkimg

            def on_key(self, event):
                'Handle keyboard event.'

                cur_sample = app.cur_sample

                if event.keysym == 'Escape':
                    root.destroy()
                    return
                if event.keysym == 'Right' or event.keysym == 'Up':
                    cur_sample += 1
                if event.keysym == 'Left' or event.keysym == 'Down':
                    cur_sample -= 1

                app.cur_sample = cur_sample % len(imgnames)

                app.update_image()

            def on_key_cur_sample(self, event):
                if event.keysym == 'Return':
                    self.update_image()
                    self.image_panel.focus_set()
                if event.keysym == 'Escape':
                    self.image_panel.focus_set()

            def on_press_save_image(self):
                if self.savable_image is None:
                    return

                index = self.ordering[self.cur_sample]

                filename = tk.filedialog.asksaveasfilename(
                    defaultextension='.png',
                    initialfile='image_{:04d}.png'.format(index))

                if filename:
                    self.savable_image.save(filename)

            def init_gui(self):
                self.master.title('Pose estimation results explorer')

                toolbar = tk.Frame(self.master)

                self.var_index = tk.StringVar()
                lbl_index = tk.Label(toolbar, width=12, textvariable=self.var_index)
                lbl_index.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

                self.var_cur_sample = tk.StringVar()
                self.var_cur_sample.set('0')
                txt_cur_sample = tk.Spinbox(toolbar,
                    textvariable=self.var_cur_sample,
                    command=self.update_image,
                    wrap=True,
                    from_=0,
                    to=len(imgnames) - 1,
                    font=tk.font.Font(size=12))
                txt_cur_sample.bind('<Key>', self.on_key_cur_sample)
                txt_cur_sample.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
                self.txt_cur_sample = txt_cur_sample

                lbl_skeleton = tk.Label(toolbar, text='Skeleton:')
                lbl_skeleton.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
                self.var_skeleton = tk.StringVar()
                self.var_skeleton.set(self.SKELETON_PREDICTION)
                option = tk.OptionMenu(toolbar, self.var_skeleton,
                    self.SKELETON_NONE, self.SKELETON_TRUTH, self.SKELETON_PREDICTION,
                    command=lambda event: self.update_image())
                option.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

                self.var_crop_as_input = tk.IntVar()
                cb_crop = tk.Checkbutton(toolbar, text='Crop',
                    variable=self.var_crop_as_input,
                    command=self.update_image)
                cb_crop.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

                btn_save = tk.Button(toolbar, text='Save image',
                    command=self.on_press_save_image)
                btn_save.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

                toolbar.pack(side=tk.TOP, fill=tk.X)

                image_panel = tk.Label(self.master)
                image_panel.configure(background='#333333')
                image_panel.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)
                image_panel.bind('<Key>', self.on_key)
                image_panel.focus_set()
                image_panel.bind('<Button-1>', lambda event: event.widget.focus_set())
                image_panel.bind('<Configure>', lambda event: self.update_image())
                self.image_panel = image_panel

                self.pack()

        root = tk.Tk()
        root.geometry("1280x720+0+0")
        app = PoseResultsFrame()

        root.update()
        app.update_image()

        root.mainloop()

if __name__ == '__main__':
    main()
