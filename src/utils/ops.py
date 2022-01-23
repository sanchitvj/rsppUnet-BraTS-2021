import random, os
import torch
import numpy as np
import pandas as pd
from pprint import pprint
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
import SimpleITK as sitk
from torch.cuda.amp import autocast

# from .metrics import calculate_metrics


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def region_wise_seg_maps(maps, threshold=0.5):

    #     inputs, pads = pad_batch1_to_compatible_size(inputs)
    #     maxz, maxy, maxx = (
    #         maps.size(2) - pads[0],
    #         maps.size(3) - pads[1],
    #         maps.size(4) - pads[2],
    #     )
    #     maps = maps[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
    #     maps = pad_or_crop_image(maps[0], target_size=(155, 240, 240))
    #     segs = np.zeros((1, 3, 155, 240, 240))
    segs = torch.zeros((1, 3, 160, 160, 160))
    segs[0, :, :, :, :] = maps[0]
    segs = segs[0] > threshold  # .numpy(); not tensor anymore

    #     print(segs.shape)
    et = segs[0]
    net = np.logical_and(segs[1], np.logical_not(et))
    ed = np.logical_and(segs[2], np.logical_not(segs[1]))

    labelmap = torch.zeros(segs[0].shape)
    labelmap[et] = 4
    labelmap[net] = 1
    labelmap[ed] = 2
    # labelmap = sitk.GetImageFromArray(labelmap)
    # labelmap = sitk.GetArrayFromImage(labelmap)
    # NOTE: print statements
    #     refmap_et, refmap_tc, refmap_wt = [torch.zeros(labelmap.shape) for i in range(3)]
    #     refmap_et = labelmap == 4
    #     refmap_tc = np.logical_or(refmap_et, labelmap == 1)
    #     refmap_wt = np.logical_or(refmap_tc, labelmap == 2)
    #     print(refmap_wt.shape)

    mask_WT = labelmap.clone()  # .copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 4] = 1

    mask_TC = labelmap.clone()  # copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 4] = 1

    mask_ET = labelmap.clone()  # copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 4] = 1
    #     et_present = 1 if np.sum(et) >= 1 else 0
    #     tc = np.logical_or(patient_label == 4, patient_label == 1)
    #     wt = np.logical_or(tc, patient_label == 2)
    #     print(mask_ET.shape)

    return labelmap, mask_ET, mask_TC, mask_WT
