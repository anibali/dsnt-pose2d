import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils import model_zoo
from dsnt.nn import DSNT

class ResNetLocalizer(nn.Module):
  def __init__(self, resnet, n_chans=1, truncate=0):
    """Create a localisation network based on ResNet

    Args:
      resnet (nn.Module): ResNet model which will form the base of the model
      n_chans (int): Number of output locations
      truncate (int): Number of ResNet layer groups to chop off
    """

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
    if truncate > 0:
      feats = resnet_groups[-truncate][0].conv1.in_channels
    else:
      feats = resnet.fc.in_features
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

def build_mpii_pose_model(base='resnet34', truncate=0):
  """Create a ResNet-based pose estimation model with pretrained parameters.

    Args:
      base (str): Base ResNet model type (eg 'resnet34')
      truncate (int): Number of ResNet layer groups to chop off
  """

  assert(base.startswith('resnet'))
  model_name = base.replace('-', '')
  builder_method = getattr(models, model_name)
  model_url = models.resnet.model_urls[model_name]

  resnet = builder_method()
  resnet.load_state_dict(model_zoo.load_url(model_url, './models'))
  model = ResNetLocalizer(resnet, n_chans=16, truncate=truncate)

  return model
