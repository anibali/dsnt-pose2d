'''
Miscellaneous utility functions.
'''

import math
import time
from contextlib import contextmanager

import torch
from PIL.ImageDraw import Draw

# Joints to connect for visualisation, giving the effect of drawing a
# basic "skeleton" of the pose.
BONES = {
    'right_lower_leg': (0, 1),
    'right_upper_leg': (1, 2),
    'right_pelvis': (2, 6),
    'left_lower_leg': (4, 5),
    'left_upper_leg': (3, 4),
    'left_pelvis': (3, 6),
    'center_lower_torso': (6, 7),
    'center_upper_torso': (7, 8),
    'center_head': (8, 9),
    'right_lower_arm': (10, 11),
    'right_upper_arm': (11, 12),
    'right_shoulder': (12, 8),
    'left_lower_arm': (14, 15),
    'left_upper_arm': (13, 14),
    'left_shoulder': (13, 8),
}

def draw_skeleton(img, coords, joint_mask=None):
    '''Draw a pose skeleton connecting joints (for visualisation purposes).

    Left-hand-side joints are connected with blue lines. Right-hand-size joints
    are connected with red lines. Center joints are connected with magenta
    lines.

    Args:
        img (PIL.Image.Image): PIL image which the skeleton will be drawn over.
        coords (Tensor): 16x2 tensor containing 0-based pixel coordinates
            of joint locations. Joints indices are expected to match
            http://human-pose.mpi-inf.mpg.de/#download
        joint_mask (Tensor, optional): Mask of valid joints (invalid joints
            will be drawn with grey lines).
    '''

    draw = Draw(img)
    for bone_name, (j1, j2) in BONES.items():
        if bone_name.startswith('center_'):
            colour = (255, 0, 255) # Magenta
        elif bone_name.startswith('left_'):
            colour = (0, 0, 255) # Blue
        elif bone_name.startswith('right_'):
            colour = (255, 0, 0) # Red
        else:
            colour = (255, 255, 255)

        if joint_mask is not None:
            # Change colour to grey if either vertex is not masked in
            if joint_mask[j1] == 0 or joint_mask[j2] == 0:
                colour = (100, 100, 100)

        draw.line([coords[j1, 0], coords[j1, 1], coords[j2, 0], coords[j2, 1]], fill=colour)

def draw_gaussian(img_tensor, x, y, sigma, normalize=False, clip_size=None):
    '''Draw a Gaussian in a single-channel 2D image.

    Args:
        img_tensor: Image tensor to draw to.
        x: x-coordinate of Gaussian centre (in pixels).
        y: y-coordinate of Gaussian centre (in pixels).
        sigma: Standard deviation of Gaussian (in pixels).
        normalize: Ensures values sum to 1 when True.
        clip_size: Restrict the size of the draw region.
    '''

    if img_tensor.dim() == 2:
        height, width = list(img_tensor.size())
    elif img_tensor.dim() == 3:
        n_chans, height, width = list(img_tensor.size())
        assert n_chans == 1, 'expected img_tensor to have one channel'
        img_tensor = img_tensor[0]
    else:
        raise Exception('expected img_tensor to have 2 or 3 dimensions')

    radius = max(width, height)
    if clip_size is not None:
        radius = clip_size / 2

    if radius < 0.5 or x <= -radius or y <= -radius or \
            x >= (width - 1) + radius or y >= (height - 1) + radius:
        return

    start_x = max(0, math.ceil(x - radius))
    end_x = min(width, int(x + radius + 1))
    start_y = max(0, math.ceil(y - radius))
    end_y = min(height, int(y + radius + 1))
    w = end_x - start_x
    h = end_y - start_y

    subimg = img_tensor[start_y:end_y, start_x:end_x]

    xs = torch.arange(start_x, end_x).type_as(img_tensor).view(1, w).expand_as(subimg)
    ys = torch.arange(start_y, end_y).type_as(img_tensor).view(h, 1).expand_as(subimg)

    k = -0.5 * (1 / sigma)**2
    subimg.copy_((xs - x)**2)
    subimg.add_((ys - y)**2)
    subimg.mul_(k)
    subimg.exp_()

    if normalize:
        val_sum = subimg.sum()
        if val_sum > 0:
            subimg.div_(val_sum)

def encode_heatmaps(coords, width, height):
    '''Convert normalised coordinates into heatmaps.'''

    # Normalised coordinates to pixel coordinates
    coords.add_(1)
    coords[:, :, 0].mul_(width / 2)
    coords[:, :, 1].mul_(height / 2)
    coords.add_(-0.5)

    batch_size = coords.size(0)
    n_chans = coords.size(1)
    target = torch.FloatTensor(batch_size, n_chans, height, width).zero_()
    for i in range(batch_size):
        for j in range(n_chans):
            x = round(coords[i, j, 0])
            y = round(coords[i, j, 1])
            draw_gaussian(target[i, j], x, y, 1, normalize=False, clip_size=7)

    return target

def decode_heatmaps(heatmaps):
    '''Convert heatmaps into normalised coordinates.'''

    batch_size, n_chans, height, width = list(heatmaps.size())

    maxval, idx = torch.max(heatmaps.view(batch_size, n_chans, -1), 2)

    maxval = maxval.view(batch_size, n_chans, 1)
    idx = idx.view(batch_size, n_chans, 1)

    coords = idx.repeat(1, 1, 2)

    coords[:, :, 0] = coords[:, :, 0] % width
    coords[:, :, 1] = coords[:, :, 1] / height

    coords = coords.float()

    # Pixel coordinates to normalised coordinates
    coords.add_(0.5)
    coords[:, :, 0].mul_(2 / width)
    coords[:, :, 1].mul_(2 / height)
    coords.add_(-1)

    # When maxval is zero, select coords (0, 0) which correspond to image centre
    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    torch.mul(coords, pred_mask, out=coords)

    return coords

@contextmanager
def timer(meter):
    start_time = time.perf_counter()
    yield
    time_elapsed = time.perf_counter() - start_time
    meter.add(time_elapsed)

def generator_timer(generator, meter):
    while True:
        with timer(meter):
            vals = next(generator)
        yield vals
