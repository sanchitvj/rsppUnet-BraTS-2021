import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch, os, random, imageio
from torch.cuda.amp import autocast
from tqdm import tqdm
import torch.nn.functional as F


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pad_batch1_to_compatible_size(batch):

    # print(batch.shape)
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


def generate_segmentations(
    data_loader, model, device, save_folder, verbose=False, visualize=False
):

    model.eval()
    for i, batch in enumerate(tqdm(data_loader, total=len(data_loader))):

        inputs = batch["image"]
        patient_id = batch["id"][0]
        ref_path = batch["seg_path"][0]

        inputs, pads = pad_batch1_to_compatible_size(inputs)

        inputs = inputs.to(device)  # cuda()

        with autocast():
            with torch.no_grad():
                pre_segs = model(inputs)

        # remove pads
        maxz, maxy, maxx = (
            pre_segs.size(2) - pads[0],
            pre_segs.size(3) - pads[1],
            pre_segs.size(4) - pads[2],
        )
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
        # print(pre_segs.shape)
        segs = torch.zeros((1, 3, 155, 240, 240))
        segs[0, :, :, :, :] = pre_segs[0]
        # segs = segs.argmax(0) # takes very long time
        # model_preds.append(segs)

        # pre_segs = torch.stack(model_preds).mean(dim=0)
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        # print("et shape: ", et.shape) # (155 ,240, 240)
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))

        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2

        if verbose:
            print(
                "1:",
                np.sum(labelmap == 1),
                " | 2:",
                np.sum(labelmap == 2),
                " | 4:",
                np.sum(labelmap == 4),
            )
            print(
                "WT:",
                np.sum((labelmap == 1) | (labelmap == 2) | (labelmap == 4)),
                " | TC:",
                np.sum((labelmap == 1) | (labelmap == 4)),
                " | ET:",
                np.sum(labelmap == 4),
            )
        # labelmap = labelmap[::-1]
        # print("labelmap: ", labelmap.shape) # (155 ,240, 240)
        labelmap_arr = labelmap.transpose(2, 1, 0)
        labelmap = sitk.GetImageFromArray(labelmap)  # .transpose(2, 1, 0))
        ref_seg_img = sitk.ReadImage(ref_path)
        # print("ref img: ", ref_seg_img.GetSize()) # (240, 240, 240)

        Direction = ref_seg_img.GetDirection()
        Origin = ref_seg_img.GetOrigin()
        Spacing = ref_seg_img.GetSpacing()

        labelmap.SetOrigin(Origin)
        labelmap.SetSpacing(Spacing)
        labelmap.SetDirection(Direction)
        labelmap.CopyInformation(ref_seg_img)
        # original mask size: (240, 240, 155)
        # print("after copy info: ", labelmap.GetSize()) # (240, 240, 240)

        # print(f"Writing {save_folder}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{save_folder}/{patient_id}.nii.gz")

        visual = f"{save_folder}/visuals"

        if visualize:
            """--- grey figure---"""
            # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
            # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
            # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
            # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
            """ --- colorful figure--- """
            Snapshot_img = np.zeros(shape=(240, 240, 3, 155), dtype=np.uint8)
            Snapshot_img[:, :, 0, :][np.where(labelmap_arr == 1)] = 255
            Snapshot_img[:, :, 1, :][np.where(labelmap_arr == 2)] = 255
            Snapshot_img[:, :, 2, :][np.where(labelmap_arr == 4)] = 255

            for frame in range(155):
                if not os.path.exists(os.path.join(visual, patient_id)):
                    os.makedirs(os.path.join(visual, patient_id))
                # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                imageio.imwrite(
                    os.path.join(visual, patient_id, str(frame) + ".png"),
                    Snapshot_img[:, :, :, frame],
                )
