import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch, os, random
import yaml
from matplotlib import pyplot as plt
from numpy import logical_and as l_and, logical_not as l_not
from torch.cuda.amp import autocast

import torch.nn.functional as F
from itertools import combinations, product

import torch

trs = list(combinations(range(2, 5), 2)) + [None]
flips = list(range(2, 5)) + [None]
rots = list(range(1, 4)) + [None]
transform_list = list(product(flips, rots))


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def simple_tta(x):
    """Perform all transpose/mirror transform possible only once.

    Sample one of the potential transform and return the transformed image and a lambda function to revert the transform
    Random seed should be set before calling this function
    """
    out = [[x, lambda z: z]]
    for flip, rot in transform_list[:-1]:
        if flip and rot:
            trf_img = torch.rot90(x.flip(flip), rot, dims=(3, 4))
            back_trf = revert_tta_factory(flip, -rot)
        elif flip:
            trf_img = x.flip(flip)
            back_trf = revert_tta_factory(flip, None)
        elif rot:
            trf_img = torch.rot90(x, rot, dims=(3, 4))
            back_trf = revert_tta_factory(None, -rot)
        else:
            raise
        out.append([trf_img, back_trf])
    return out


def apply_simple_tta(model, x, average=True):
    todos = simple_tta(x)
    out = []
    for im, revert in todos:
        if model.deep_supervision:
            out.append(revert(model(im)[0]).sigmoid_().cpu())
        else:
            out.append(revert(model(im)).sigmoid_().cpu())
    if not average:
        return out
    return torch.stack(out).mean(dim=0)


def revert_tta_factory(flip, rot):
    if flip and rot:
        return lambda x: torch.rot90(x.flip(flip), rot, dims=(3, 4))
    elif flip:
        return lambda x: x.flip(flip)
    elif rot:
        return lambda x: torch.rot90(x, rot, dims=(3, 4))
    else:
        raise


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


def generate_segmentations(data_loader, model, args, save_folder):

    for i, batch in enumerate(data_loader):

        # measure data loading time
        inputs = batch["image"]
        patient_id = batch["id"][0]
        ref_path = batch["seg_path"][0]
        #         print(patient_id)

        inputs, pads = pad_batch1_to_compatible_size(inputs)

        inputs = inputs.cuda()

        with autocast():
            with torch.no_grad():
                pre_segs = model(inputs)
                # pre_segs = torch.sigmoid(pre_segs)

        # remove pads
        maxz, maxy, maxx = (
            pre_segs.size(2) - pads[0],
            pre_segs.size(3) - pads[1],
            pre_segs.size(4) - pads[2],
        )
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
        segs = torch.zeros((1, 3, 155, 240, 240))
        segs[0, :, :, :, :] = pre_segs[0]
        # model_preds.append(segs)

        # pre_segs = torch.stack(model_preds).mean(dim=0)
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))

        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_seg_img = sitk.ReadImage(ref_path)
        labelmap.CopyInformation(ref_seg_img)

        #         print(f"Writing {save_folder}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{save_folder}/{patient_id}.nii.gz")
