'''
Code for building neural network models.
'''

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision import models

from dsnt.nn import DSNT, euclidean_loss
from dsnt import util, hourglass

class HumanPoseModel(nn.Module):
    '''Abstract base class for human pose estimation models.'''

    def forward_loss(self, out_var, target_var, mask_var):
        '''Calculate the value of the loss function.'''
        raise NotImplementedError()

    def compute_coords(self, out_var):
        '''Calculate joint coordinates from the network output.'''
        raise NotImplementedError()

class ResNetHumanPoseModel(HumanPoseModel):
    '''Create a ResNet-based model for human pose estimation.

    Args:
        resnet (nn.Module): ResNet model which will form the base of the model
        n_chans (int): Number of output locations
        truncate (int): Number of ResNet layer groups to chop off
    '''

    def __init__(self, resnet, n_chans=16, truncate=0):
        super().__init__()

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
        if truncate > 0:
            feats = resnet_groups[-truncate][0].conv1.in_channels
        else:
            feats = resnet.fc.in_features
        self.hm_conv = nn.Conv2d(feats, self.n_chans, kernel_size=1, bias=False)
        self.hm_preact = nn.Softmax()
        self.hm_dsnt = DSNT()
        self.out_size = 7 * (2 ** truncate)

        self.input_size = 224

    def forward_loss(self, out_var, target_var, mask_var):
        loss = euclidean_loss(out_var, target_var, mask_var)
        return loss

    def compute_coords(self, out_var):
        return out_var.data.type(torch.FloatTensor)

    def forward(self, *inputs):
        x = inputs[0]
        x = self.fcn(x)
        x = self.hm_conv(x)
        x = x.view(-1, self.out_size*self.out_size)
        x = self.hm_preact(x)
        x = x.view(-1, self.n_chans, self.out_size, self.out_size)
        x = self.hm_dsnt(x)
        return x

class HourglassHumanPoseModel(HumanPoseModel):
    def __init__(self, hg, n_chans=16):
        super().__init__()

        self.hg = hg
        self.n_chans = n_chans

        self.input_size = 256

    def forward_loss(self, out_var, target_var, mask_var):
        norm_coords = target_var.data.cpu()
        width = out_var[0].size(-1)
        height = out_var[0].size(-2)

        target_hm = util.encode_heatmaps(norm_coords, width, height)
        target_hm_var = Variable(target_hm.cuda())

        # Calculate and sum up intermediate losses
        loss = sum([nn.functional.mse_loss(hm, target_hm_var) for hm in out_var])

        return loss

    def compute_coords(self, out_var):
        return util.decode_heatmaps(out_var[-1].data.cpu())

    def forward(self, *inputs):
        x = inputs[0]

        # Zero-center input so pixel range is [-0.5, 0.5]
        x = x - 0.5

        x = self.hg(x)
        return x

def build_mpii_pose_model(base='resnet34', truncate=0):
    '''Create a ResNet-based pose estimation model with pretrained parameters.

        Args:
            base (str): Base ResNet model type (eg 'resnet34')
            truncate (int): Number of ResNet layer groups to chop off
    '''

    if base.startswith('resnet'):
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

        model = ResNetHumanPoseModel(resnet, n_chans=16, truncate=truncate)
    elif base.startswith('hg'):
        if base == 'hg1':
            hg = hourglass.hg1()
        elif base == 'hg2':
            hg = hourglass.hg2()
        elif base == 'hg4':
            hg = hourglass.hg4()
        elif base == 'hg8':
            hg = hourglass.hg8()
        else:
            raise Exception('unsupported base model type: ' + base)

        model = HourglassHumanPoseModel(hg, n_chans=16)
    else:
        raise Exception('unsupported base model type: ' + base)

    return model
