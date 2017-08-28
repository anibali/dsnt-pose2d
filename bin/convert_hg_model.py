#!/usr/bin/env python3

'''
Code for converting a pretrained model from Bearpaw's PyTorch port of
Stacked Hourglass networks into a model usable in our project.

https://github.com/bearpaw/pytorch-pose
'''

from collections import OrderedDict
import torch

import dsnt.hourglass
import dsnt.model

def main():
    '''Main conversion entrypoint function.'''

    in_file = 'models/hg1.pth'
    out_file = 'models/complete/hg-s1-b1.pth'
    stacks = 1
    blocks = 1

    in_state = torch.load(in_file)
    model_desc = {
        'base': 'hg',
        'stacks': stacks,
        'blocks': blocks,
    }

    old_state_dict = in_state['state_dict']

    # Remove the "module." prefix from key names
    new_state_dict = OrderedDict([(k[7:], v) for (k, v) in old_state_dict.items()])

    model = dsnt.model.build_mpii_pose_model(**model_desc)
    model.hg.load_state_dict(new_state_dict)

    out_state = {
        'epoch': in_state['epoch'],
        'optimizer': in_state['optimizer'],
        'model_desc': model_desc,
        'state_dict': model.state_dict()
    }

    torch.save(out_state, out_file)

if __name__ == '__main__':
    main()
