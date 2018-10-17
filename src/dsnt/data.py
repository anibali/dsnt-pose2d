'''
Dataset loaders.
'''

import random

import math
import numpy as np
import torch
import torch.nn.functional
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchdata.mpii import MpiiData, MPII_Joint_Horizontal_Flips, MPII_Image_Mean, \
    MPII_Image_Stddev, transform_keypoints


class ImageSpecs():
    def __init__(self, size, subtract_mean, divide_stddev):
        self._size = size
        self._subtract_mean = subtract_mean
        self._divide_stddev = divide_stddev

    @property
    def size(self):
        return self._size

    @property
    def subtract_mean(self):
        return self._subtract_mean

    @property
    def divide_stddev(self):
        return self._divide_stddev

    def convert(self, img, dataset_stats):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)

        # Scale the image down
        img = nn.functional.adaptive_avg_pool2d(img, self.size).data

        if self.subtract_mean:
            mean = dataset_stats.MEAN
        else:
            mean = [0, 0, 0]
        if self.divide_stddev:
            stddev = dataset_stats.STDDEV
        else:
            stddev = [1, 1, 1]
        img = transforms.Normalize(mean, stddev)(img)

        return img

    def unconvert(self, img_tensor, dataset_stats):
        """Convert a Torch tensor into a PIL image, unapplying specs.

        Args:
            img_tensor (torch.FloatTensor):
            dataset_stats:

        Returns:
            Image.Image:
        """

        img_tensor = img_tensor.clone()

        if self.subtract_mean:
            mean = dataset_stats.MEAN
        else:
            mean = [0, 0, 0]
        if self.divide_stddev:
            stddev = dataset_stats.STDDEV
        else:
            stddev = [1, 1, 1]

        for t, m, s in zip(img_tensor, mean, stddev):
            t.mul_(s).add_(m)

        img = transforms.ToPILImage()(img_tensor)
        return img


class MPIIDataset(Dataset):
    '''Create a Dataset object for loading MPII Human Pose data.

    Args:
        data_dir: path to the MPII dataset directory
        subset: subset of the data to load ("train", "val", or "test")
        use_aug: set to `True` to enable random data augmentation
        size: resolution of the input images
    '''

    # This tensor describes how to rearrange joint indices in the case of a
    # horizontal flip transformation.
    HFLIP_INDICES = torch.LongTensor(MPII_Joint_Horizontal_Flips)

    # Per-channel mean and standard deviation values for the dataset
    # Channel order: red, green, blue
    MEAN = MPII_Image_Mean
    STDDEV = MPII_Image_Stddev

    def __init__(self, data_dir, subset='train', use_aug=False,
                 image_specs=ImageSpecs(224, False, False), max_length=None):
        super().__init__()

        self.subset = subset
        self.use_aug = use_aug
        self.image_specs = image_specs

        self.mpii_data = MpiiData(data_dir)
        self.example_ids = self.mpii_data.subset_indices(self.subset)[:max_length]

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        id = self.example_ids[index]

        # The resolution of the subject bounding box when reading the image
        src_res = 384

        raw_image = self.mpii_data.load_cropped_image(id, size=src_res, margin=int(src_res / 4))
        normalize = float(self.mpii_data.head_lengths[id])
        matrix = self.mpii_data.get_bb_transform(id)
        part_coords = torch.from_numpy(transform_keypoints(self.mpii_data.keypoints[id], matrix))
        part_mask = torch.from_numpy(self.mpii_data.keypoint_masks[id])

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
        if hflip:
            # Mirror x coordinate (horizontal flip)
            t = torch.mm(t.new([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]), t)
        # Scale then rotate
        rads = math.radians(rot)
        t = torch.mm(t.new([
            [math.cos(rads) / scale, math.sin(rads) / scale, 0],
            [-math.sin(rads) / scale, math.cos(rads) / scale, 0],
            [0, 0, 1],
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
            # image bounds.
            #
            # NOTE: It is still possible for joints to be transformed outside of the image bounds
            # when augmentations are turned off. This is because the center/scale information
            # provided in the MPII human pose dataset will occasionally not include a joint.
            # For example, see the head top joint for ID 21 in the validation set.
            if self.subset == 'train':
                within_bounds, _ = part_coords.abs().lt(1).min(-1, keepdim=False)
                part_mask.mul_(within_bounds)

        ### Set up transforms for returning to original image coords ###

        # NOTE: Care has to be taken when converting from model output to original
        # coordinate space in the case of horizontal flips, since the transform
        # matrices can't rearrange joints for the user. It is up to the user to
        # check "hflip" and flip the joints appropriately.

        s = torch.mm(torch.from_numpy(np.linalg.inv(matrix)), torch.inverse(t))
        trans_m = s[0:2, 0:2]
        trans_b = s[0:2, 2].contiguous().view(1, 2)

        ### Transform image ###

        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if hflip else img),
            transforms.Lambda(lambda img: img.rotate(rot, Image.BILINEAR) if rot != 0 else img),
            transforms.CenterCrop(src_res * scale),
            transforms.ToTensor(),
        ])
        input_image = trans(raw_image)

        # Colour augmentation
        # * bearpaw/pytorch-pose uses uniform(0.8, 1.2)
        # * anewell/pose-hg-train uses uniform(0.6, 1.4)
        if self.use_aug:
            for chan in range(input_image.size(0)):
                input_image[chan].mul_(random.uniform(0.6, 1.4)).clamp_(0, 1)

        input_image = self.image_specs.convert(input_image, self)

        ### Return the sample ###

        sample = {
            'normalize': normalize,
            'transform_b': trans_b,
            'transform_m': trans_m,
            'input': input_image,
            'part_mask': part_mask,
            'part_coords': part_coords.float(),
            'hflip': hflip,
        }

        return sample
