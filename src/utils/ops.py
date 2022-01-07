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


def determinist_collate(batch):
    batch = pad_batch_to_max_shape(batch)
    return default_collate(batch)


def pad_batch1_to_compatible_size(batch):
    #     print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(2), ymax - batch.size(3), xmax - batch.size(4)
    assert all(
        pad >= 0 for pad in (zpad, ypad, xpad)
    ), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)


def pad_batch_to_max_shape(batch):
    shapes = (sample["mask"].shape for sample in batch)
    _, z_sizes, y_sizes, x_sizes = list(zip(*shapes))
    maxs = [int(max(z_sizes)), int(max(y_sizes)), int(max(x_sizes))]
    for i, max_ in enumerate(maxs):
        max_stride = 16
        if max_ % max_stride != 0:
            # Make it divisible by 16
            maxs[i] = ((max_ // max_stride) + 1) * max_stride
    zmax, ymax, xmax = maxs
    for elem in batch:
        exple = elem["mask"]
        zpad, ypad, xpad = (
            zmax - exple.shape[1],
            ymax - exple.shape[2],
            xmax - exple.shape[3],
        )
        assert all(
            pad >= 0 for pad in (zpad, ypad, xpad)
        ), "Negative padding value error !!"
        # free data augmentation
        left_zpad, left_ypad, left_xpad = [
            random.randint(0, pad) for pad in (zpad, ypad, xpad)
        ]
        right_zpad, right_ypad, right_xpad = [
            pad - left_pad
            for pad, left_pad in zip(
                (zpad, ypad, xpad), (left_zpad, left_ypad, left_xpad)
            )
        ]
        pads = (left_xpad, right_xpad, left_ypad, right_ypad, left_zpad, right_zpad)
        elem["image"], elem["mask"] = F.pad(
            torch.from_numpy(elem["image"]), pads
        ), F.pad(torch.from_numpy(elem["mask"]), pads)
    return batch


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)


def pad_or_crop_image(image, seg=None, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [
        get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))
    ]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [
        get_left_right_idx_should_pad(size, dim)
        for size, dim in zip(target_size, [z, y, x])
    ]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image


def region_wise_seg_maps(inputs, maps, threshold=0.5):

    #     inputs, pads = pad_batch1_to_compatible_size(inputs)
    #     maxz, maxy, maxx = (
    #         maps.size(2) - pads[0],
    #         maps.size(3) - pads[1],
    #         maps.size(4) - pads[2],
    #     )
    #     maps = maps[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
    #     maps = pad_or_crop_image(maps[0], target_size=(155, 240, 240))
    #     segs = np.zeros((1, 3, 155, 240, 240))
    maps[0, :, :, :, :] = maps  # [0]
    segs = maps[0] > threshold  # .numpy(); not tensor anymore

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
    #     print("labelmap: ", labelmap.shape)

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

    #     print(mask_ET.shape)

    return labelmap, mask_ET, mask_TC, mask_WT
