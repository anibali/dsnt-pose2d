#!/usr/bin/env python3

"""
This script will print out some info about a model.
"""

import argparse
import subprocess
import torch
from torch.autograd import Variable

from dsnt.model import build_mpii_pose_model


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='DSNT human pose model info')
    parser.add_argument(
        '--model', type=str, metavar='PATH', required=True,
        help='model state file')
    parser.add_argument(
        '--gpu', type=int, metavar='N', default=0,
        help='index of the GPU to use')

    args = parser.parse_args()

    return args


def  get_gpu_used_memory():
    nvidia_smi_out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
    mem_usage = [int(line) for line in nvidia_smi_out.splitlines(keepends=False)]
    return mem_usage


def main():
    """Main model info entrypoint function."""

    args = parse_args()

    torch.cuda.set_device(args.gpu)
    old_mem_usage = get_gpu_used_memory()[torch.cuda.current_device()]

    model_file = args.model

    model_state = torch.load(model_file)
    model = build_mpii_pose_model(**model_state['model_desc'])
    model.load_state_dict(model_state['state_dict'])
    model.cuda(torch.cuda.current_device())

    print(model_state['model_desc'])

    param_count = 0

    for param in model.parameters():
        param_count += param.data.numel()

    print('Number of parameters: {:0.2f} million'.format(param_count / 1e6))

    input_size = model.image_specs.size

    print('Expected input size: {:d} pixels'.format(input_size))

    dummy_data = torch.cuda.FloatTensor(8, 3, input_size, input_size).uniform_(0, 1)

    model.train()
    out_var = model(Variable(dummy_data))
    if isinstance(out_var, list):
        out_var = out_var[-1]
    out_var.sum().backward()

    new_mem_usage = get_gpu_used_memory()[torch.cuda.current_device()]
    print('Training memory usage for a batch of size 8: {:0.0f} MiB'.format(
        new_mem_usage - old_mem_usage))

    del dummy_data


if __name__ == '__main__':
    main()
