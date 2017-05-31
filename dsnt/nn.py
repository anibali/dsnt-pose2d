import torch
import torch.nn as nn
from torch.autograd import Variable

class DSNT(nn.Module):
  """The DSNT layer takes n-channel spatial input and outputs n pairs of
  coordinates. Works with and without batches.

  For example, let's say you are trying to create a network for finding the
  coordinates of some object in an image. The first part of the network can be a
  normal CNN, but at the end you reduce the number of channels to 1 and apply this
  module as the final layer.

  Input: Spatial heatmap data (batches x n x rows x cols)
  Output: Numerical coordinates (batches x n x 2).
          Top-left = (x=-1, y=-1), bottom-right = (x=1, y=1).
  """

  def __init__(self):
    super().__init__()
    self.fixed_weights = None

  # Prepare the fixed weight matrix used in the forward and backward passes
  def prepare_fixed_weights(self, width, height, tensor_new):
    # Return early if the fixed_weights matrix already exists and is valid
    if self.fixed_weights is not None and width == self.width and height == self.height:
      return self.fixed_weights
    self.width = width
    self.height = height

    # Define coordinates of first and last pixel centers such that
    # the top-left of the first pixel is (-1, -1) and the bottom-right of the
    # last pixel is (1, 1)
    first_x = -(width - 1) / width
    first_y = -(height - 1) / height
    last_x = (width - 1) / width
    last_y = (height - 1) / height

    # Create the fixed_weights matrix
    if self.fixed_weights:
      w = self.fixed_weights.data
    else:
      w = tensor_new()
    w.resize_(2, width*height)

    # Populate the weights
    weights_view = w.view(2, height, width)
    x_weights = torch.linspace(first_x, last_x, width)
    y_weights = torch.linspace(first_y, last_y, height)
    weights_view[0].copy_(x_weights.view(1, width).expand(height, width))
    weights_view[1].copy_(y_weights.view(height, 1).expand(height, width))

    self.fixed_weights = Variable(w, requires_grad=False)
    return self.fixed_weights

  def forward(self, x):
    if x.dim() == 3:
      batch_mode = False
      batch_size = 1
      n_chans, height, width = list(x.size())
    elif x.dim() == 4:
      batch_mode = True
      batch_size, n_chans, height, width = list(x.size())
    else:
      raise 'DSNT expects 3D or 4D input'

    fixed_weights = self.prepare_fixed_weights(width, height, x.data.new)

    x_view = x.view(batch_size*n_chans, height*width)
    output = x_view.mm(fixed_weights.transpose(0, 1))

    # Set the appropriate view for output
    output_size = batch_mode and [batch_size, n_chans, 2] or [n_chans, 2]

    return output.view(output_size)

class EuclideanLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input, target, mask=None):
    if input.dim() == 2:
      batch_mode = False
      batch_size = 1
      n_chans, coord_dim = list(input.size())
    elif input.dim() == 3:
      batch_mode = True
      batch_size, n_chans, coord_dim = list(input.size())
    else:
      raise 'EuclideanLoss expects 2D or 3D input'

    # Calculate Euclidean distances between input and target locations
    diff = input - target
    diff_sq = diff * diff
    dist_sq = diff_sq.sum(input.dim() - 1)
    dist = dist_sq.sqrt()

    # Apply mask to distances
    if mask is not None:
      dist = dist * mask

    # Divide loss to make it independent of batch size
    loss = dist.sum() / (batch_size * n_chans)

    return loss

__all__ = ['DSNT', 'EuclideanLoss']
