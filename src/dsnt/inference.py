import progressbar
import torch
from tele.meter import SumMeter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchdata.mpii import MpiiData

from dsnt.data import MPIIDataset
from dsnt.util import timer, type_as_index, reverse_tensor


def generate_predictions(model, dataset, use_flipped=True, batch_size=1, time_meter=None):
    """Generate predictions with the model"""

    if use_flipped:
        assert batch_size == 1, 'test-time flip augmentation only work with batch_size=1'

    sum_meter = SumMeter()

    model.cuda()
    model.eval()

    loader = DataLoader(dataset, batch_size, num_workers=4, pin_memory=True)
    preds = torch.DoubleTensor(len(dataset), 16, 2).zero_()

    completed = 0
    with progressbar.ProgressBar(max_value=len(dataset)) as bar:
        for i, batch in enumerate(loader):
            batch_size = batch['input'].size(0)
            sum_meter.reset()

            with timer(sum_meter):
                if use_flipped:
                    sample = batch['input']
                    rev_sample = reverse_tensor(batch['input'], -1)
                    in_var = Variable(torch.cat([sample, rev_sample], 0).cuda(), volatile=True)

                    hm_var = model.forward_part1(in_var)
                    if isinstance(hm_var, list):
                        # Just use the last heatmap from stacked hourglass
                        hm_var = hm_var[-1]
                    hm1, hm2 = hm_var.split(1)
                    hm2 = reverse_tensor(hm2, -1)
                    hm2 = hm2.index_select(-3, type_as_index(MPIIDataset.HFLIP_INDICES, hm2))

                    hm = (hm1 + hm2) / 2
                    out_var = model.forward_part2(hm)
                    coords = model.compute_coords(out_var)
                else:
                    in_var = Variable(batch['input'].cuda(), volatile=True)
                    out_var = model(in_var)
                    coords = model.compute_coords(out_var)

                orig_preds = torch.baddbmm(
                    batch['transform_b'],
                    coords.double(),
                    batch['transform_m'])

            pos = i * batch_size
            preds[pos:(pos + batch_size)] = orig_preds

            if time_meter is not None:
                time_meter.add(sum_meter.value())

            completed += batch_size
            bar.update(completed)

    return preds


def evaluate_mpii_predictions(preds, subset, evaluator):
    mpii_data = MpiiData('/datasets/mpii')

    subset_indices = mpii_data.subset_indices(subset)
    actual = torch.from_numpy(mpii_data.keypoints[subset_indices])
    head_lengths = torch.from_numpy(mpii_data.head_lengths[subset_indices])
    joint_mask = torch.from_numpy(mpii_data.keypoint_masks[subset_indices])

    # Calculate PCKh accuracies
    evaluator.add(preds, actual, joint_mask, head_lengths)

    return evaluator
