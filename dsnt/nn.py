"""
Custom reusable nn modules.
"""

from functools import reduce
from operator import mul
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable, Function


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
        self.width = None
        self.height = None

    # Prepare the fixed weight matrix used in the forward and backward passes
    def _prepare_fixed_weights(self, width, height, tensor_new):
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

    def forward(self, *inputs):
        x = inputs[0]

        if x.dim() == 3:
            batch_mode = False
            batch_size = 1
            n_chans, height, width = list(x.size())
        elif x.dim() == 4:
            batch_mode = True
            batch_size, n_chans, height, width = list(x.size())
        else:
            raise Exception('DSNT expects 3D or 4D input')

        fixed_weights = self._prepare_fixed_weights(width, height, x.data.new)

        x_view = x.view(batch_size*n_chans, height*width)
        output = x_view.mm(fixed_weights.transpose(0, 1))

        # Set the appropriate view for output
        if batch_mode:
            output_size = [batch_size, n_chans, 2]
        else:
            output_size = [n_chans, 2]

        return output.view(output_size)


def dsnt(input, square_coords=False):
    *first_dims, height, width = input.size()

    first_x = -(width - 1) / width
    first_y = -(height - 1) / height
    last_x = (width - 1) / width
    last_y = (height - 1) / height

    sing_dims = [1] * len(first_dims)
    xs = torch.linspace(first_x, last_x, width).view(*sing_dims, 1, width)
    ys = torch.linspace(first_y, last_y, height).view(*sing_dims, height, 1)

    if isinstance(input, Variable):
        xs = Variable(xs, requires_grad=False)
        ys = Variable(ys, requires_grad=False)

    xs = xs.type_as(input)
    ys = ys.type_as(input)

    if square_coords:
        xs = xs ** 2
        ys = ys ** 2

    output_xs = (input * xs).view(*first_dims, height * width).sum(-1, keepdim=False)
    output_ys = (input * ys).view(*first_dims, height * width).sum(-1, keepdim=False)
    output = torch.stack([output_xs, output_ys], -1)

    return output


def _avg_losses(losses, mask=None):
    if mask is not None:
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom


def euclidean_loss(actual, target, mask=None):
    """Calculate the average Euclidean loss for multi-point samples.

    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).

    Args:
        actual (Tensor): Predictions ([batches x] n x d)
        target (Tensor): Ground truth target ([batches x] n x d)
        mask (Tensor, optional): Mask of points to include in the loss calculation
            ([batches x] n), defaults to including everything
    """

    # Calculate Euclidean distances between actual and target locations
    diff = actual - target
    diff_sq = diff * diff
    dist_sq = diff_sq.sum(actual.dim() - 1)
    dist = dist_sq.sqrt()

    return _avg_losses(dist, mask)


class ThresholdedSoftmax(Function):
    """

    """

    @staticmethod
    def forward(ctx, inp, threshold=-np.inf, eps=1e-12):
        mask = inp.ge(threshold).type_as(inp)

        d = -inp.max(-1, keepdim=True)[0]
        exps = (inp + d).exp() * mask
        out = exps / (exps.sum(-1, keepdim=True) + eps)

        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        [out] = ctx.saved_variables

        # Same as normal softmax gradient calculation
        sum = (grad_output * out).sum(-1, keepdim=True)
        grad_input = out * (grad_output - sum)

        return grad_input, None, None


def thresholded_softmax(inp, threshold=-np.inf, eps=1e-12):
    """A softmax variant which masks out inputs below a certain threshold.

    For the normal softmax operation, all outputs will be greater than
    zero. In contrast, this softmax variant ensures that inputs below
    the given threshold value will result in a corresponding zero in the
    output. The output will still be a valid probability distribution
    (sums to 1).

    Args:
        inp: The tensor containing input activations
        threshold: The threshold value applied to input activations
        eps: A small number to prevent division by zero
    """

    return ThresholdedSoftmax.apply(inp, threshold, eps)


def make_gauss(coords, width, height, sigma):
    first_x = -(width - 1) / width
    first_y = -(height - 1) / height
    last_x = (width - 1) / width
    last_y = (height - 1) / height

    sing_dims = [1] * (coords.dim() - 1)
    xs = torch.linspace(first_x, last_x, width).view(*sing_dims, 1, width).expand(*sing_dims, height, width)
    ys = torch.linspace(first_y, last_y, height).view(*sing_dims, height, 1).expand(*sing_dims, height, width)

    if isinstance(coords, Variable):
        xs = Variable(xs, requires_grad=False)
        ys = Variable(ys, requires_grad=False)

    xs = xs.type_as(coords)
    ys = ys.type_as(coords)

    k = -0.5 * (1 / sigma)**2
    xs = (xs - coords.narrow(-1, 0, 1).unsqueeze(-1)) ** 2
    ys = (ys - coords.narrow(-1, 1, 1).unsqueeze(-1)) ** 2
    gauss = ((xs + ys) * k).exp()

    # Normalize the Gaussians
    val_sum = gauss.sum(-1, keepdim=True).sum(-2, keepdim=True) + 1e-24
    gauss = gauss / val_sum

    return gauss


def _kl_2d(p, q, eps=1e-24):
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = unsummed_kl.sum(-1, keepdim=False).sum(-1, keepdim=False)
    return kl_values


def _js_2d(p, q, eps=1e-24):
    m = 0.5 * (p + q)
    return 0.5 * _kl_2d(p, m, eps) + 0.5 * _kl_2d(q, m, eps)


def kl_gauss_2d(inp, coords, mask=None, sigma=1):
    gauss = make_gauss(coords, inp.size(-1), inp.size(-2), sigma)
    kls = _kl_2d(inp, gauss)

    return _avg_losses(kls, mask)


def js_gauss_2d(inp, coords, mask=None, sigma=1):
    gauss = make_gauss(coords, inp.size(-1), inp.size(-2), sigma)
    kls = _js_2d(inp, gauss)

    return _avg_losses(kls, mask)


def mse_gauss_2d(inp, coords, mask=None, sigma=1):
    gauss = make_gauss(coords, inp.size(-1), inp.size(-2), sigma)
    sq_error = (inp - gauss) ** 2
    sq_error_sum = sq_error.sum(-1, keepdim=False).sum(-1, keepdim=False)

    return _avg_losses(sq_error_sum, mask)
