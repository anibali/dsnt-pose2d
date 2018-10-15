import tkinter as tk
import tkinter.font
import tkinter.filedialog

from enum import Enum
import h5py
import os
from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms
from torch.autograd import Variable
from PIL import ImageTk, Image
from dsnt.util import draw_skeleton
from dsnt.data import MPIIDataset
from dsnt.evaluator import PCKhEvaluator


class SortBy(Enum):
    ID = 1
    ACCURACY = 2


@lru_cache(maxsize=32)
def generate_heatmaps(model, image_file, scale, center):
    img = Image.open(image_file)
    orig_width = img.width
    orig_height = img.height

    # Calculate crop used for input
    cx, cy = center
    size = scale * 200
    crop_box = [cx - size / 2, cy - size / 2,
                cx + size / 2, cy + size / 2]
    crop_box = [round(x) for x in crop_box]
    img = img.crop(crop_box)
    img = img.resize((model.image_specs.size, model.image_specs.size), Image.BILINEAR)

    print('Running model on: {}'.format(image_file))
    img_tensor = model.image_specs.convert(img, MPIIDataset)
    img_tensor = img_tensor.unsqueeze(0).type(torch.cuda.FloatTensor)
    model(Variable(img_tensor, volatile=True))
    hms_tensor = model.heatmaps.data.cpu()[0]

    heatmaps = []

    for hm_tensor in hms_tensor.split(1, 0):
        # Scale and clamp pixel values
        hm_tensor.div_(hm_tensor.max()).clamp_(0, 1)
        # Convert tensor to PIL Image
        hm_img = torchvision.transforms.ToPILImage()(hm_tensor)
        hm_img = hm_img.resize((round(size), round(size)), Image.NEAREST)
        # "Uncrop" heatmap to match original image size
        hm_padded = Image.new('RGB', (orig_width, orig_height), (0, 0, 0))
        hm_padded.paste(hm_img, (crop_box[0], crop_box[1]))
        # Add heatmap to list
        heatmaps.append(hm_padded)

    return heatmaps


class PoseResultsFrame(tk.Frame):
    SKELETON_NONE = 'None'
    SKELETON_TRUTH = 'Ground truth'
    SKELETON_PREDICTION = 'Prediction'

    def __init__(self, image_dir, sort_by, preds, annot_file, model=None):
        super().__init__()

        with h5py.File(annot_file, 'r') as f:
            self.actual = torch.from_numpy(f['part'][:])
            self.image_names = [name.decode('utf-8') for name in f['imgname'][:]]
            self.head_lengths = torch.from_numpy(f['normalize'][:])
            self.centers = torch.from_numpy(f['center'][:])
            self.scales = torch.from_numpy(f['scale'][:])
        self.joint_mask = self.actual.select(2, 0).gt(1).mul(self.actual.select(2, 1).gt(1))

        self.image_files = [os.path.join(image_dir, name) for name in self.image_names]
        self.preds = preds
        self.model = model

        if sort_by == SortBy.ACCURACY:
            n_joints = preds.size(1)
            scores = []

            # Compute interesting orderings
            for b in range(preds.size(0)):
                dists = []
                for j in range(n_joints):
                    if self.joint_mask[b, j] == 1:
                        dist = torch.dist(self.actual[b, j], preds[b, j]) / self.head_lengths[b]
                        dists.append(dist)
                if len(dists) > 0:
                    scores.append(sum(dists) / len(dists))
                else:
                    scores.append(0)
            self.ordering = list(np.argsort(np.array(scores))[::-1])
        else:
            self.ordering = list(range(preds.size(0)))

        self.savable_image = None

        self.init_gui()

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

    @property
    def show_heatmap(self):
        return self.var_show_heatmap.get() == 1

    def update_image(self):
        index = self.ordering[self.cur_sample]
        self.var_index.set('Index: {:04d}'.format(index))

        if self.show_heatmap:
            img = self.get_joint_heatmap()
        else:
            img = Image.open(self.image_files[index])

        if self.var_skeleton.get() == self.SKELETON_TRUTH:
            draw_skeleton(img, self.actual[index], self.joint_mask[index])
        elif self.var_skeleton.get() == self.SKELETON_PREDICTION:
            draw_skeleton(img, self.preds[index], self.joint_mask[index])

        if self.crop_as_input:
            # Calculate crop used for input
            scale = self.scales[index]
            center = self.centers[index].clone()
            size = scale * 200

            img = img.crop([center[0] - size / 2, center[1] - size / 2,
                            center[0] + size / 2, center[1] + size / 2])

        self.savable_image = img.copy()

        width = self.image_panel.winfo_width()
        height = self.image_panel.winfo_height() - 2
        img.thumbnail((width, height), Image.ANTIALIAS)
        tkimg = ImageTk.PhotoImage(img)
        self.image_panel.configure(image=tkimg)
        self.image_panel.image = tkimg

    def on_key(self, event):
        """Handle keyboard event."""

        cur_sample = self.cur_sample

        if event.keysym == 'Escape':
            self.master.destroy()
            return
        if event.keysym == 'Right':
            cur_sample += 1
        if event.keysym == 'Left':
            cur_sample -= 1
        if event.keysym == 'Home':
            cur_sample = 0
        if event.keysym == 'End':
            cur_sample = len(self.image_files) - 1

        self.cur_sample = cur_sample % len(self.image_files)

        self.update_image()

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

    def get_joint_heatmap(self):
        joint_id = PCKhEvaluator.JOINT_NAMES.index(self.var_joint.get())

        index = self.ordering[self.cur_sample]
        heatmaps = generate_heatmaps(
            self.model, self.image_files[index], self.scales[index], tuple(self.centers[index]))

        return heatmaps[joint_id].copy()

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
                                    to=len(self.image_files) - 1,
                                    font=tk.font.Font(size=12))
        txt_cur_sample.bind('<Key>', self.on_key_cur_sample)
        txt_cur_sample.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.txt_cur_sample = txt_cur_sample

        lbl_skeleton = tk.Label(toolbar, text='Skeleton:')
        lbl_skeleton.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.var_skeleton = tk.StringVar()
        self.var_skeleton.set(self.SKELETON_PREDICTION)
        opt_skeleton = tk.OptionMenu(
            toolbar, self.var_skeleton, self.SKELETON_NONE, self.SKELETON_TRUTH,
            self.SKELETON_PREDICTION, command=lambda event: self.update_image())
        opt_skeleton.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        self.var_crop_as_input = tk.IntVar()
        cb_crop = tk.Checkbutton(toolbar, text='Crop',
                                 variable=self.var_crop_as_input,
                                 command=self.update_image)
        cb_crop.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        btn_save = tk.Button(toolbar, text='Save image',
                             command=self.on_press_save_image)
        btn_save.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        lbl_joint = tk.Label(toolbar, text='Heatmap joint:')
        lbl_joint.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        self.var_joint = tk.StringVar()
        self.var_joint.set(PCKhEvaluator.JOINT_NAMES[0])
        opt_joint = tk.OptionMenu(
            toolbar, self.var_joint, *PCKhEvaluator.JOINT_NAMES,
            command=lambda event: self.update_image())
        opt_joint.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)

        self.var_show_heatmap = tk.IntVar()
        cb_hm = tk.Checkbutton(toolbar, text='Show heatmap',
                                 variable=self.var_show_heatmap,
                                 command=self.update_image)
        cb_hm.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
        if self.model is None:
            cb_hm['state'] = tk.DISABLED

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


def run_gui(preds, annot_file, model=None):
    # Directory containing the original MPII human pose image JPEGs
    image_dir = '/data/dlds/cache/mpii-human-pose/images'

    root = tk.Tk()
    root.geometry("1280x720+0+0")
    app = PoseResultsFrame(
        image_dir, SortBy.ID, preds, annot_file, model)

    root.update()
    app.update_image()

    root.mainloop()
