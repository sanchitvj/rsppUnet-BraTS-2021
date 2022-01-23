import yaml, random, os
import torch
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
from medpy import metric
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
from .ops import region_wise_seg_maps
from .ops import pad_batch1_to_compatible_size


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]


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


def save_metrics(
    epoch, metrics, swa, writer, current_epoch, teacher=False, save_folder=None
):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(
        f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
        [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()],
    )
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(
            f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
            [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()],
            file=f,
        )
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{'_swa' if swa else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


def calculate_metrics(preds, targets, patient=None, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    # pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]
    #     print(targets[0].shape)
    targets, preds = targets[0], preds[0]
    metrics_list = []
    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if torch.sum(targets[i]) == 0:
            # print(f"{label} not present for {patient}")
            #             sens = np.nan
            dice = torch.tensor(1) if torch.sum(preds[i]) == 0 else torch.tensor(0)
        #             tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
        #             fp = np.sum(l_and(preds[i], l_not(targets[i])))
        #             spec = tn / (tn + fp)
        #             haussdorf_dist = np.nan

        else:
            #             preds_coords = np.argwhere(preds[i])
            #             targets_coords = np.argwhere(targets[i])
            #             haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            #             tp = np.sum(l_and(preds[i], targets[i]))
            #             tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            #             fp = np.sum(l_and(preds[i], l_not(targets[i])))
            #             fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            #             sens = tp / (tp + fn)
            #             spec = tn / (tn + fp)

            #             dice = 2 * tp / (2 * tp + fp + fn)
            eps = 1e-8
            intersection = torch.sum(torch.mul(preds[i], targets[i])) + eps / 2
            union = torch.sum(preds[i]) + torch.sum(targets[i]) + eps
            dice = 2 * intersection / union
        #         metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        #         metrics[SENS] = sens
        #         metrics[SPEC] = spec
        # pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


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
    #     batch_size = labels.size(0)
    # y_pred = outputs[:, 0, :, :, :]
    # y_truth = targets[:, 0, :, :, :]
    #     print(logits.shape, labels.shape)
    logits = logits > threshold
    logits = logits.type(torch.FloatTensor)
    intersection = torch.sum(torch.mul(logits, labels)) + eps / 2
    union = torch.sum(logits) + torch.sum(labels) + eps
    dice = 2 * intersection / union

    return dice


def dice_coefficient_1(logits, labels, threshold=0.5, eps=1e-8):
    # outputs, targets = outputs.to("cpu"), targets.to("cpu")
    #     batch_size = labels.size(0)
    y_pred = logits[:, 0, :, :, :]
    y_truth = labels[:, 0, :, :, :]
    #     print("below d1... ")
    #     print(logits.shape, labels.shape)
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps / 2
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union

    return dice.mean()
