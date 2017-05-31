import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os.path as path
import h5py, h5py_cache
import numpy as np
import random
import math

class MPIIDataset(Dataset):
  HFLIP_INDICES = torch.LongTensor([
    5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10
  ])

  def __init__(self, data_dir, subset='train', use_aug=False):
    super().__init__()

    h5_file = path.join(data_dir, 'mpii-human-pose.h5')
    self.h5_file = h5_file
    self.subset = subset
    self.use_aug = use_aug
    with h5py_cache.File(h5_file, 'r', chunk_cache_mem_size=1024**3) as f:
      self.length = f[subset]['images'].shape[0]

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    subset = self.subset

    with h5py_cache.File(self.h5_file, 'r', chunk_cache_mem_size=1024**3) as f:
      raw_image = torch.from_numpy(f[subset]['images'][index])
      trans_m = torch.from_numpy(f[subset]['transforms/m'][index]).double()
      trans_b = torch.from_numpy(f[subset]['transforms/b'][index]).double()
      normalize = f[subset]['normalize'][index]

      if 'parts' in f[subset]:
        part_coords = torch.from_numpy(f[subset]['parts/coords'][index])
        norm_target = part_coords.double()
        orig_target = torch.mm(norm_target, trans_m).add_(trans_b.expand_as(norm_target))
        part_mask = orig_target[:, 0].gt(1).mul(orig_target[:, 1].gt(1))
      else:
        part_coords = None
        part_mask = torch.ByteTensor(16, 2).fill_(1)

    ### Calculate augmentations ###

    scale = 1
    rot = 0
    hflip = False
    if self.use_aug:
      scale = 2 ** np.clip(random.normalvariate(0, 0.25), -0.5, 0.5)
      if random.uniform(0, 1) < 0.4:
        rot = np.clip(random.normalvariate(0, 30), -60, 60)
      hflip = random.choice([False, True])

    ### Build transformation matrix ###

    t = torch.eye(3).double()
    # Zero-center
    t = torch.mm(t.new([
      [1, 0, -549 / 2],
      [0, 1, -549 / 2],
      [0, 0, 1       ],
    ]), t)
    if hflip:
      # Mirror x coordinate (horizontal flip)
      t = torch.mm(t.new([
        [-1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1],
      ]), t)
    # Scale then rotate
    rads = math.radians(rot)
    t = torch.mm(t.new([
      [math.cos(rads) / scale,  math.sin(rads) / scale, 0],
      [-math.sin(rads) / scale, math.cos(rads) / scale, 0],
      [0,                      0,                     1],
    ]), t)
    # Normalize to [-1, 1] range
    t = torch.mm(t.new([
      [2/383, 0,     0],
      [0,     2/383, 0],
      [0,     0,     1],
    ]), t)

    ### Transform joint coords ###

    if part_coords is not None:
      coords = torch.DoubleTensor(part_coords.size(0), 3)
      coords[:, 0:2].copy_(part_coords)
      coords[:, 2].fill_(1)
      part_coords.copy_(torch.mm(coords, t.transpose(0, 1))[:, 0:2])

      if hflip:
        # Swap left and right joints
        hflip_indices_2d = MPIIDataset.HFLIP_INDICES.view(-1, 1).expand_as(part_coords)
        part_coords.scatter_(0, hflip_indices_2d, part_coords.clone())
        part_mask.scatter_(0, MPIIDataset.HFLIP_INDICES, part_mask.clone())

      # Mask out joints that have been transformed to a location outside of the
      # image bounds
      for i in range(part_coords.size(0)):
        x, y = part_coords[i].tolist()
        if x < -1 or x > 1 or y < -1 or y > 1:
          part_mask[i] = 0

    ### Transform image ###

    # I'm pretty unhappy about this. I prepared the input data into
    # CxHxW layout because this is what Torch uses, but I have to convert
    # to and from HxWxC using ToPILImage and ToTensor. Why aren't there
    # image transforms available which operate on tensors directly?
    trans = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if hflip else img),
      transforms.Lambda(lambda img: img.rotate(rot, Image.BILINEAR) if rot != 0 else img),
      transforms.CenterCrop(384 * scale),
      transforms.Scale(224, Image.BILINEAR),
      transforms.ToTensor(),
    ])
    input_image = trans(raw_image)

    # Colour augmentation
    if self.use_aug:
      for chan in range(input_image.size(0)):
        input_image[chan].mul_(random.uniform(0.6, 1.4)).clamp_(0, 1)

    ### Set up transforms for returning to original image coords ###

    # NOTE: Care has to be taken when converting from model output to original
    # coordinate space in the case of horizontal flips, since the transform
    # matrices can't rearrange joints for the user. It is up to the user to
    # check "hflip" and flip the joints appropriately.

    u = torch.eye(3).double()
    u[0:2, 0:2].copy_(trans_m)
    u[0:2, 2].copy_(trans_b)
    s = torch.mm(u, torch.inverse(t))
    trans_m.copy_(s[0:2, 0:2])
    trans_b.copy_(s[0:2, 2])

    ### Return the sample ###

    sample = {
      'normalize':    normalize,
      'transform_b':  trans_b,
      'transform_m':  trans_m,
      'input':        input_image,
      'part_mask':    part_mask,
      'hflip':        hflip,
    }
    if part_coords is not None:
      sample['part_coords'] = part_coords

    return sample
