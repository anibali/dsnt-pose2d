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
        # # Pairs of joints used in MPII evaluation code
        # # * https://github.com/anibali/eval-mpii-pose/blob/2eaa57babc353d1604c31d91583ad79ca0484d5b/eval/genTablePCK.m#L5
        # 'ankle': {'rankle', 'lankle'},
        # 'knee': {'rknee', 'lknee'},
        # 'hip': {'rhip', 'lhip'},
        # 'wrist': {'rwrist', 'lwrist'},
        # 'elbow': {'relbow', 'lelbow'},
        # 'shoulder': {'rshoulder', 'lshoulder'},
        # 'head': {'headtop', 'upperneck'},

        # "Upper body" according to calculations in the MPII evaluation code
        # * https://github.com/anibali/eval-mpii-pose/blob/2eaa57babc353d1604c31d91583ad79ca0484d5b/eval/annolist2matrix.m#L22
        # * https://github.com/anibali/eval-mpii-pose/blob/2eaa57babc353d1604c31d91583ad79ca0484d5b/eval/computePCK.m#L14-L18
        'ubody': {'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow', 'lwrist'},

        # "Total accuracy" according to calculations by anewell and bearpaw
        # * https://github.com/anewell/pose-hg-train/blob/2fef6915fbd836a5d218a5d2f0c87c463532f1a6/src/util/dataset/mpii.lua#L6
        # * https://github.com/bearpaw/pytorch-pose/blob/3e3e6debde71fee93ef37f7936ccbd5ad0925b33/example/mpii.py#L28
        'total_anewell': {'rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle',
                        'rwrist', 'relbow', 'lelbow', 'lwrist'},

        # "Total accuracy" according to calculations in the MPII evaluation code
        # * https://github.com/anibali/eval-mpii-pose/blob/2eaa57babc353d1604c31d91583ad79ca0484d5b/eval/annolist2matrix.m#L21-L23
        'total_mpii': set(JOINT_NAMES) - {'pelvis', 'thorax'},

        # Total accuracy for all of the joints
        'all': set(JOINT_NAMES),
    }

    def __init__(self, threshold=0.5):
        self.threshold = threshold

        meter_names = self.JOINT_NAMES + list(self.JOINT_GROUPS.keys())
        meters = {name: AverageValueMeter() for name in meter_names}

        meters_for_joints = {}
        for j, meter_name in enumerate(self.JOINT_NAMES):
            meters_for_joints[j] = {meters[meter_name]}
        for meter_name, joint_names in self.JOINT_GROUPS.items():
            for joint_name in joint_names:
                j = self.JOINT_NAMES.index(joint_name)
                meters_for_joints[j].add(meters[meter_name])

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
