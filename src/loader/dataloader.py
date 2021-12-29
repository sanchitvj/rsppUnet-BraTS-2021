import os, pathlib, time, random
import gzip
from posixpath import dirname
from albumentations import augmentations
from cv2 import phase, phaseCorrelate
from numpy.core.fromnumeric import size
from numpy.lib.type_check import imag
from tqdm import tqdm

from skimage.transform import resize  # not using, highly inefficient
from skimage.util import montage
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import Compose, HorizontalFlip, RandomResizedCrop
from albumentations.pytorch import ToTensorV2

from augmentations import DataAugmentation


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


class BratsDataset(Dataset):
    def __init__(
        self,
        args,
        data,
        phase: str = "train",
    ):

        self.data = data
        self.phase = phase
        self.augmentations = args.augs
        self.size = args.dataset.img_size
        self.teacher_model = args.dataset.teacher_model
        if self.teacher_model:
            self.data_types = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
            self.seg_type = "_seg.nii"
        else:
            self.data_types = [
                "_flair.nii.gz",
                "_t1.nii.gz",
                "_t1ce.nii.gz",
                "_t2.nii.gz",
            ]
            self.seg_type = "_seg.nii.gz"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        root_path = self.data[idx]
        dir_name = root_path.split("/")[-1]

        # load all modalities
        images = []
        for data_type in self.data_types:
            img_name = dir_name + data_type
            img_path = os.path.join(root_path, img_name)
            img = self.load_img(img_path)  # .transpose(2, 0, 1)
            img = self.normalize(img)
            images.append(img)

        img = np.stack(images)
        # FIXME: preprocessing removed; collate not working.
        # Remove maximum extent of the zero-background to make future crop more useful
        #        z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(img, axis=0) != 0)
        # Add 1 pixel in each side
        #        zmin, ymin, xmin = [
        #            max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)
        #        ]
        #       zmax, ymax, xmax = [
        #          int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)
        #     ]
        #    img = img[:, zmin:zmax, ymin:ymax, xmin:xmax]

        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        mask_name = dir_name + self.seg_type
        mask_path = os.path.join(root_path, mask_name)
        mask = self.load_img(mask_path)

        mask = self.preprocess_mask_labels(mask)
        # mask = mask[:, zmin:zmax, ymin:ymax, xmin:xmax]

        # FIXME: not using pad_crop for validation gives error(spp_net3D)
        img, mask = pad_or_crop_image(img, mask, target_size=self.size)
        if self.phase == "train":

            if self.augmentations.apply:
                augment = DataAugmentation(self.augmentations)
                img, mask = augment(img.astype(np.float32), mask.astype(np.float32))
                # mask = augment(mask.astype(np.float32))
        #                 img, mask = pad_or_crop_image(img, mask, target_size=self.size)
        else:
            img, mask = img.astype(np.float32), mask.astype(np.float32)

        return {
            # INFO: https://stackoverflow.com/questions/57517740/pytorch-custom-dataset-valueerror-some-of-the-strides-of-a-given-numpy-array-a
            # "image": torch.from_numpy(img.copy()),
            # "mask": torch.from_numpy(mask.copy()),
            "image": img,
            "mask": mask,
        }

    def load_img(self, file_path):
        img = nib.load(file_path)
        data = np.asarray(img.dataobj)
        return data

    def normalize(self, data: np.ndarray, low_perc=1, high_perc=99):
        """Main pre-processing function used for the challenge (seems to work the best).
        Remove outliers voxels first, then min-max scale.
        Warnings
        --------
        This will not do it channel wise!!
        """

        non_zeros = data > 0
        low, high = np.percentile(data[non_zeros], [low_perc, high_perc])
        image = np.clip(data, low, high)

        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


def get_dataset(
    args,
    fold_num,
    n_splits=5,
):
    train_dir = []
    for filename in os.listdir(args.dataset.data_path):
        f = os.path.join(args.dataset.data_path, filename)
        train_dir.append(f)

    kfold = KFold(n_splits, shuffle=True, random_state=args.seed)
    splits = list(kfold.split(train_dir))
    train_idx, val_idx = splits[fold_num]
    train = [train_dir[i] for i in train_idx]
    val = [train_dir[i] for i in val_idx]

    train_dataset = BratsDataset(args, train, phase="train")
    val_dataset = BratsDataset(args, val, phase="val")
    return train_dataset, val_dataset


# class dataset:

#     data_path = "/home/sanchit/Segmentation Research/BraTS Data/loader_test"
#     teacher_model = False  # True
#     num_workers = 4  # batch_size * ngpu
#     batch_size = 1
#     img_size = (16, 16, 16)


# class augs:

#     apply = True
#     aug_name = None  # flip, rotate, shift, brightness, elastic_transform
#     aug_type = "multiple"  # robust
#     aug_prob = 0.75
#     min_angle = -30
#     max_angle = 30
#     max_percentage = 0.4


# class args:
#     # data_path = "/home/sanchit/Segmentation Research/BraTS Data/loader_test"

#     # data_path = data_path
#     seed = 1235
#     # img_size = (32, 32, 32)
#     # augs = dict(apply=True)
#     # teacher_model = False
#     # min_angle = -30
#     # max_angle = 30
#     # max_percentage = 0.4
#     dataset = dataset
#     augs = augs


# if __name__ == "__main__":

#     start1 = time.time()
#     for i in range(1):
#         start2 = time.time()
#         train_dataset, val_dataset = get_dataset(args, fold_num=i)

#         train_loader = DataLoader(
#             train_dataset, batch_size=1, shuffle=True, num_workers=4
#         )
#         val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

#         loader = tqdm(train_loader, total=len(train_loader))
#         for step, batch in enumerate(tqdm(loader)):
#             if i % 4 == 0:
#                 print(step, batch["image"].shape)
#                 pass
#         print(f"time taken for fold {i}: {time.time() - start2}")
#     print("total time: ", (time.time() - start1))
