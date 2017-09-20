"""
Code for building neural network models.
"""

import inspect
import re

import torch
from torch import nn
import torch.nn.functional
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision import models

from dsnt.nn import DSNT, euclidean_loss, thresholded_softmax
from dsnt import util, hourglass, minihg
from dsnt.data import ImageSpecs


class HumanPoseModel(nn.Module):
    """Abstract base class for human pose estimation models."""

    def _hm_preact(self, x, preact):
        n_chans = x.size(-3)
        height = x.size(-2)
        width = x.size(-1)
        x = x.view(-1, height * width)
        if preact == 'softmax':
            x = nn.functional.softmax(x)
        elif preact == 'thresholded_softmax':
            x = thresholded_softmax(x, -0.5)
        else:
            raise Exception('unrecognised heatmap preactivation function: {}'.format(preact))
        x = x.view(-1, n_chans, height, width)
        return x

    @property
    def image_specs(self):
        """Specifications of expected input images."""
        raise NotImplementedError()

    def forward_loss(self, out_var, target_var, mask_var):
        """Calculate the value of the loss function."""
        raise NotImplementedError()

    def compute_coords(self, out_var):
        """Calculate joint coordinates from the network output."""
        raise NotImplementedError()


class ResNetHumanPoseModel(HumanPoseModel):
    """Create a ResNet-based model for human pose estimation.

    Args:
        resnet (nn.Module): ResNet model which will form the base of the model
        n_chans (int): Number of output locations
        dilate (int): Number of ResNet layer groups to use dilation for instead of downsampling
        truncate (int): Number of ResNet layer groups to chop off
        output_strat (str): Strategy for going between heatmaps and coords (dsnt, fc, gauss)
    """

    def __init__(self, resnet, n_chans=16, dilate=0, truncate=0, output_strat='dsnt', preact='softmax'):
        super().__init__()

        self.n_chans = n_chans
        self.output_strat = output_strat
        self.preact = preact

        self.heatmap_size = 7 * 2 ** max(dilate, truncate)

        fcn_modules = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        ]
        layers = [resnet.layer2, resnet.layer3, resnet.layer4]

        for i, layer in enumerate(layers[len(layers) - dilate:]):
            dilx = dily = 2 ** (i + 1)
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    if module.stride == (2, 2):
                        module.stride = (1, 1)
                    elif module.kernel_size == (3, 3):
                        kx, ky = module.kernel_size
                        module.dilation = (dilx, dily)
                        module.padding = ((dilx * (kx - 1) + 1) // 2, (dily * (ky - 1) + 1) // 2)

        fcn_modules.extend(layers[:len(layers)-truncate])
        self.fcn = nn.Sequential(*fcn_modules)
        if truncate > 0:
            feats = layers[-truncate][0].conv1.in_channels
        else:
            feats = resnet.fc.in_features
        self.hm_conv = nn.Conv2d(feats, self.n_chans, kernel_size=1, bias=False)

        if self.output_strat == 'dsnt':
            self.hm_dsnt = DSNT()
        elif self.output_strat == 'fc':
            self.out_fc = nn.Linear(self.heatmap_size * self.heatmap_size, 2)

    @property
    def image_specs(self):
        return ImageSpecs(size=224, subtract_mean=False, divide_stddev=False)

    def forward_loss(self, out_var, target_var, mask_var):
        if self.output_strat == 'dsnt' or self.output_strat == 'fc':
            loss = euclidean_loss(out_var, target_var, mask_var)
            return loss
        elif self.output_strat == 'gauss':
            norm_coords = target_var.data.cpu()
            width = out_var.size(-1)
            height = out_var.size(-2)

            target_hm = util.encode_heatmaps(norm_coords, width, height)
            target_hm_var = Variable(target_hm.cuda())

            loss = nn.functional.mse_loss(out_var, target_hm_var)
            return loss

        raise Exception('invalid configuration')

    def compute_coords(self, out_var):
        if self.output_strat == 'dsnt' or self.output_strat == 'fc':
            return out_var.data.type(torch.FloatTensor)
        elif self.output_strat == 'gauss':
            return util.decode_heatmaps(out_var.data.cpu())

        raise Exception('invalid configuration')

    def forward(self, *inputs):
        x = inputs[0]
        x = self.fcn(x)
        x = self.hm_conv(x)

        if self.output_strat == 'dsnt':
            x = self._hm_preact(x, self.preact)
            self.heatmaps = x
            x = self.hm_dsnt(x)
        elif self.output_strat == 'fc':
            x = self._hm_preact(x, self.preact)
            self.heatmaps = x
            height = x.size(-2)
            width = x.size(-1)
            x = x.view(-1, height * width)
            x = self.out_fc(x)
            x = x.view(-1, self.n_chans, 2)
        else:
            self.heatmaps = x

        return x


class HourglassHumanPoseModel(HumanPoseModel):
    def __init__(self, hg, n_chans=16, output_strat='gauss', preact='softmax'):
        super().__init__()

        self.hg = hg
        self.n_chans = n_chans
        self.output_strat = output_strat
        self.preact = preact
        try:
            self.heatmap_size = hg.heatmap_size
        except AttributeError:
            self.heatmap_size = 64

        if self.output_strat == 'dsnt':
            self.hm_dsnt = DSNT()
        elif self.output_strat == 'fc':
            self.out_fc = nn.Linear(self.heatmap_size * self.heatmap_size, 2)

    @property
    def image_specs(self):
        return ImageSpecs(size=256, subtract_mean=True, divide_stddev=False)

    def forward_loss(self, out_var, target_var, mask_var):
        if self.output_strat == 'dsnt' or self.output_strat == 'fc':
            # Calculate and sum up intermediate losses
            loss = sum([euclidean_loss(hm, target_var, mask_var) for hm in out_var])

            return loss
        elif self.output_strat == 'gauss':
            norm_coords = target_var.data.cpu()
            width = out_var[0].size(-1)
            height = out_var[0].size(-2)

            target_hm = util.encode_heatmaps(norm_coords, width, height)
            target_hm_var = Variable(target_hm.cuda())

            # Calculate and sum up intermediate losses
            loss = sum([nn.functional.mse_loss(hm, target_hm_var) for hm in out_var])

            return loss

        raise Exception('invalid configuration')

    def compute_coords(self, out_var):
        last_out_var = out_var[-1]

        if self.output_strat == 'dsnt' or self.output_strat == 'fc':
            return last_out_var.data.type(torch.FloatTensor)
        elif self.output_strat == 'gauss':
            return util.decode_heatmaps(last_out_var.data.cpu())

        raise Exception('invalid configuration')

    def forward(self, *inputs):
        x = inputs[0]

        hg_outs = self.hg(x)
        out = []

        if self.output_strat == 'gauss':
            self.heatmaps = hg_outs[-1]
            out = hg_outs
        elif self.output_strat == 'dsnt':
            for x in hg_outs:
                x = self._hm_preact(x, self.preact)
                self.heatmaps = x
                x = self.hm_dsnt(x)
                out.append(x)
        elif self.output_strat == 'fc':
            for x in hg_outs:
                x = self._hm_preact(x, self.preact)
                self.heatmaps = x
                height = x.size(-2)
                width = x.size(-1)
                x = x.view(-1, height * width)
                x = self.out_fc(x)
                x = x.view(-1, self.n_chans, 2)
                out.append(x)
        else:
            raise Exception('invalid configuration')

        return out


def _build_resnet_pose_model(base, dilate=0, truncate=0, output_strat='dsnt', preact='softmax'):
    """Create a ResNet-based pose estimation model with pretrained parameters.

        Args:
            base (str): Base ResNet model type (eg 'resnet34')
            truncate (int): Number of ResNet layer groups to chop off
            output_strat (str): Output strategy
    """

    if base == 'resnet18':
        resnet = models.resnet18()
        model_url = models.resnet.model_urls['resnet18']
    elif base == 'resnet34':
        resnet = models.resnet34()
        model_url = models.resnet.model_urls['resnet34']
    elif base == 'resnet50':
        resnet = models.resnet50()
        model_url = models.resnet.model_urls['resnet50']
    elif base == 'resnet101':
        resnet = models.resnet101()
        model_url = models.resnet.model_urls['resnet101']
    elif base == 'resnet152':
        resnet = models.resnet152()
        model_url = models.resnet.model_urls['resnet152']
    else:
        raise Exception('unsupported base model type: ' + base)

    # Download pretrained weights (cache in the "models/" directory)
    pretrained_weights = model_zoo.load_url(model_url, './models')
    # Load pretrained weights into the ResNet model
    resnet.load_state_dict(pretrained_weights)

    model = ResNetHumanPoseModel(
        resnet, n_chans=16, dilate=dilate, truncate=truncate, output_strat=output_strat,
        preact=preact)
    return model


def _build_hg_model(base, stacks=2, blocks=1, output_strat='gauss', preact='softmax'):
    m = re.search('hg(\d+)', base)

    if m is not None:
        stacks = int(m.group(1))
    elif base == 'hg':
        pass
    else:
        raise Exception('unsupported base model type: ' + base)

    hg = hourglass.HourglassNet(hourglass.Bottleneck, num_stacks=stacks, num_blocks=blocks)

    model = HourglassHumanPoseModel(hg, n_chans=16, output_strat=output_strat, preact=preact)
    return model


def _build_minihg_model(base, stacks=2, blocks=1, depth=3, output_strat='dsnt', preact='softmax'):
    m = re.search('minihg(\d+)', base)

    if m is not None:
        stacks = int(m.group(1))
    elif base == 'minihg':
        pass
    else:
        raise Exception('unsupported base model type: ' + base)

    hg = minihg.MiniHourglassNet(depth=depth, num_stacks=stacks, num_blocks=blocks)

    model = HourglassHumanPoseModel(hg, n_chans=16, output_strat=output_strat, preact=preact)
    return model


def build_mpii_pose_model(base='resnet34', **kwargs):
    """Create a pose estimation model"""

    if base.startswith('resnet'):
        build_model = _build_resnet_pose_model
    elif base.startswith('hg'):
        build_model = _build_hg_model
    elif base.startswith('minihg'):
        build_model = _build_minihg_model
    else:
        raise Exception('unsupported base model type: ' + base)

    # Filter out unexpected parameters
    func_params = inspect.signature(build_model).parameters.values()
    param_names = [p.name for p in func_params if p.default != inspect.Parameter.empty]
    kwargs = {k: kwargs[k] for k in param_names if k in kwargs}

    return build_model(base, **kwargs)
