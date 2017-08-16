'''
Human pose metric evaluation code.
'''

import torch
from torchnet.meter import AverageValueMeter

class PCKhEvaluator:
    '''Class for calculating and accumulating PCKh values.'''

    JOINT_NAMES = [
        'rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle', 'pelvis', 'thorax',
        'upperneck', 'headtop', 'rwrist', 'relbow', 'rshoulder', 'lshoulder',
        'lelbow', 'lwrist',
    ]

    def __init__(self):
        self.meters = {
            'all': AverageValueMeter(),
        }
        for joint_name in PCKhEvaluator.JOINT_NAMES:
            self.meters[joint_name] = AverageValueMeter()

    def add(self, pred, target, joint_mask, head_lengths):
        '''Calculate and accumulate PCKh values for batch.'''

        batch_size = pred.size(0)
        n_joints = pred.size(1)

        for b in range(batch_size):
            for j in range(n_joints):
                if joint_mask[b, j] == 1:
                    dist = torch.dist(target[b, j], pred[b, j]) / head_lengths[b]
                    thresholded = dist <= 0.5 and 1 or 0
                    self.meters['all'].add(thresholded)
                    self.meters[PCKhEvaluator.JOINT_NAMES[j]].add(thresholded)

    def reset(self):
        '''Reset accumulated values to zero.'''

        for meter in self.meters.values():
            meter.reset()
