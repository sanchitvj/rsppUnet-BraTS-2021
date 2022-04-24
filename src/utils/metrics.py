import torch
import numpy as np
from medpy import metric
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def combine_labels(labels):
    """
    Combine wt, tc, et into WT; tc, et into TC; et into ET
    :param labels: torch.Tensor of size (bs, 3, ?,?,?); ? is the crop size
    :return:
    """
    whole_tumor = labels[:, :3, :, :, :].sum(1)  # could have 2 or 3
    tumor_core = labels[:, 1:3, :, :, :].sum(1)
    enhanced_tumor = labels[:, 2:3, :, :, :].sum(1)
    whole_tumor[whole_tumor != 0] = 1
    tumor_core[tumor_core != 0] = 1
    enhanced_tumor[enhanced_tumor != 0] = 1
    return whole_tumor, tumor_core, enhanced_tumor  # (bs, ?, ?, ?)


def dice_coefficient_single_label(y_pred, y_truth, eps=1e-8):
    # batch_size = y_pred.size(0)
    intersection = (
        torch.sum(torch.mul(y_pred, y_truth), dim=(-3, -2, -1)) + eps / 2
    )  # axis=?, (bs, 1)
    union = (
        torch.sum(y_pred, dim=(-3, -2, -1)) + torch.sum(y_truth, dim=(-3, -2, -1)) + eps
    )  # (bs, 1)
    dice = 2 * intersection / union
    return dice.mean()


def dice_per_region(logits, labels):

    #     predmaps, _, _, _ = region_wise_seg_maps(logits)  # pred_et, pred_tc, pred_wt
    #     labelmaps, _, _, _ = region_wise_seg_maps(labels)  # label_et, label_tc, label_wt

    #     pred_wt, label_wt = torch.gt(predmaps, 0).float(), torch.gt(labelmaps, 0).float()
    #     pred_tc, label_tc = (torch.eq(predmaps, 1) | torch.eq(predmaps, 4)).float(), (
    #         torch.eq(labelmaps, 1) | torch.eq(labelmaps, 4)
    #     ).float()
    #     pred_et, label_et = torch.eq(predmaps, 4).float(), torch.eq(labelmaps, 4).float()
    # #     print(pred_et.shape, label_et.shape)
    y_pred = logits[:, :3, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = labels[:, :3, :, :, :]
    y_pred = y_pred > 0.5  # threshold
    y_pred = y_pred.type(torch.FloatTensor)

    pred_wt, pred_tc, pred_et = combine_labels(y_pred)
    truth_wt, truth_tc, truth_et = combine_labels(y_truth)

    dice_wt = dice_coefficient_single_label(pred_wt, truth_wt)
    dice_tc = dice_coefficient_single_label(pred_tc, truth_tc)
    dice_et = dice_coefficient_single_label(pred_et, truth_et)

    hd_wt = hd_dist_per_region(pred_wt.detach().numpy(), truth_wt.detach().numpy())
    hd_tc = hd_dist_per_region(pred_tc.detach().numpy(), truth_tc.detach().numpy())
    hd_et = hd_dist_per_region(pred_et.detach().numpy(), truth_et.detach().numpy())

    return dice_et, dice_tc, dice_wt, hd_et, hd_tc, hd_wt


def hd_dist_per_region(preds, targets):

    # https://loli.github.io/medpy/_modules/medpy/metric/binary.html
    # preds_coords = np.argwhere(preds)
    # targets_coords = np.argwhere(targets)
    # haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

    # checking if voxels are empty.
    if not np.any(preds) or np.all(preds) or not np.any(targets) or np.all(targets):
        haussdorf_dist = 0
    else:
        haussdorf_dist = metric.hd95(preds, targets)  # , voxel_spacing, connectivity)

    return haussdorf_dist


# from NVnet
def dice_coefficient(logits, labels, threshold=0.5, eps=1e-8):
    # outputs, targets = outputs.to("cpu"), targets.to("cpu")
    # batch_size = labels.size(0)
    y_pred = logits[:, 0, :, :, :]
    y_truth = labels[:, 0, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps / 2
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union

    return dice.mean()
