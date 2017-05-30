import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils import model_zoo
from dsnt.nn import DSNT

ResNet34_URL = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'

class ResNetLocalizer(nn.Module):
  def __init__(self, resnet, n_chans=1, truncate=0):
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
    feats = fcn_modules[-1][0].conv1.out_channels
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

def build_mpii_pose_model(base='resnet-34', truncate=0):
  assert(base == 'resnet-34')
  resnet34_model = models.resnet34()
  resnet34_model.load_state_dict(model_zoo.load_url(ResNet34_URL, './models'))
  model = ResNetLocalizer(resnet34_model, n_chans=16, truncate=truncate)
  return model
