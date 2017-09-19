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

    JOINT_GROUPS = {
        # 'ankle': ['rankle', 'lankle'],
        # 'knee': ['rknee', 'lknee'],
        # 'hip': ['rhip', 'lhip'],
        # 'wrist': ['rwrist', 'lwrist'],
        # 'elbow': ['relbow', 'lelbow'],
        # 'shoulder': ['rshoulder', 'lshoulder'],
        # 'head': ['headtop', 'upperneck'],
        'all_hard': ['rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle',
                     'rwrist', 'relbow', 'lelbow', 'lwrist'],
        'all': JOINT_NAMES
    }

    def __init__(self, threshold=0.5):
        self.threshold = threshold

        meter_names = self.JOINT_NAMES + list(self.JOINT_GROUPS.keys())
        meters = {name: AverageValueMeter() for name in meter_names}

        meters_for_joints = {}
        for j, meter_name in enumerate(self.JOINT_NAMES):
            meters_for_joints[j] = [meters[meter_name]]
        for meter_name, joint_names in self.JOINT_GROUPS.items():
            for joint_name in joint_names:
                j = self.JOINT_NAMES.index(joint_name)
                meters_for_joints[j].append(meters[meter_name])

        self.meters = meters
        self._meters_for_joints = meters_for_joints

    @staticmethod
    def calculate_pckh_distance(pred, target, ref_dist):
        return torch.dist(target, pred) / ref_dist

    def add(self, pred, target, joint_mask, head_lengths):
        '''Calculate and accumulate PCKh values for batch.'''

        batch_size = pred.size(0)
        n_joints = pred.size(1)

        for b in range(batch_size):
            for j in range(n_joints):
                if joint_mask[b, j] == 1:
                    dist = self.calculate_pckh_distance(target[b, j], pred[b, j], head_lengths[b])
                    thresholded = 1 if dist <= self.threshold else 0
                    for meter in self._meters_for_joints[j]:
                        meter.add(thresholded)

    def reset(self):
        '''Reset accumulated values to zero.'''

        for meter in self.meters.values():
            meter.reset()
