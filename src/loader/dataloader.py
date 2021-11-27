import os, pathlib, time, random
import gzip
from posixpath import dirname
from cv2 import phaseCorrelate
from numpy.core.fromnumeric import size
from tqdm import tqdm

from skimage.transform import resize  # not using, highly inefficient
from skimage.util import montage
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt

# import h5py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import Compose, HorizontalFlip, RandomResizedCrop
from albumentations.pytorch import ToTensorV2


def get_augmentations(phase, IMG_SIZE):
    list_transforms = []

    list_trfms = Compose(list_transforms)
    return list_trfms


def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
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
        data,
        size,
        phase: str = "test",
    ):

        self.data = data
        self.phase = phase
        self.augmentations = get_augmentations(phase, size)
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
        self.size = size

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

        # Remove maximum extent of the zero-background to make future crop more useful
        z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(img, axis=0) != 0)
        # Add 1 pixel in each side
        zmin, ymin, xmin = [
            max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)
        ]
        zmax, ymax, xmax = [
            int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)
        ]
        img = img[:, zmin:zmax, ymin:ymax, xmin:xmax]
        # if self.is_resize:
        #     img = self.resize(img, self.size)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != "test":
            mask_name = dir_name + "_seg.nii.gz"
            mask_path = os.path.join(root_path, mask_name)
            mask = self.load_img(mask_path)

            mask = self.preprocess_mask_labels(mask)
            mask = mask[:, zmin:zmax, ymin:ymax, xmin:xmax]

            # if self.is_resize:
            #     mask = self.resize(mask, self.size)
            mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
            mask = np.clip(mask, 0, 1)

            # TODO: Test this
            img, mask = pad_or_crop_image(img, mask, target_size=self.size)
            # TODO add augmentations
            augmented = self.augmentations(
                image=img.astype(np.float32), mask=mask.astype(np.float32)
            )

            img = augmented["image"]
            mask = augmented["mask"]

            return {
                "image": img,
                "mask": mask,
            }

        return {
            "image": img,
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

    def resize(self, data: np.ndarray, size):
        data = resize(data, size, preserve_range=True)
        return data

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
    data_path,
    seed,
    size,
    fold_num,
    debug,
    n_splits,
):
    train_dir = []
    for filename in os.listdir(data_path):
        f = os.path.join(data_path, filename)
        train_dir.append(f)

    kfold = KFold(n_splits, shuffle=True, random_state=seed)
    splits = list(kfold.split(train_dir))
    train_idx, val_idx = splits[fold_num]
    # print("first idx of train", train_idx[0])
    # print("first idx of test", val_idx[0])
    train = [train_dir[i] for i in train_idx]
    val = [train_dir[i] for i in val_idx]

    train_dataset = BratsDataset(train, size, phase="train")
    val_dataset = BratsDataset(val, size, phase="val")
    return train_dataset, val_dataset


# if __name__ == "__main__":

#     train_dir = "/home/sanchit/Segmentation Research/BraTS Data/loader_test/"  # "../data/brats21/BraTS_2021_training"
#     start1 = time.time()
#     for i in range(2):
#         start2 = time.time()
#         train_dataset, val_dataset = get_dataset(train_dir, 1111, fold_num=i)

#         train_loader = DataLoader(
#             train_dataset, batch_size=1, shuffle=True, num_workers=4
#         )
#         val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
#         # train_data = next(iter(train_loader))
#         # val_data = next(iter(val_loader))
#         loader = tqdm(train_loader, total=len(train_loader))
#         for step, batch in enumerate(loader):
#             if i % 400 == 0:
#                 print(step, batch["image"].shape)
#         print(f"time taken for fold {i}: {time.time() - start2}")
#     print("total time: ", (time.time() - start1))
# print(len(train_loader), len(val_loader))
# print(train_data["image"][0].shape, val_data["image"][0].shape)
