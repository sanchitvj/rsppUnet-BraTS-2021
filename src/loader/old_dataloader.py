import os
import gzip

from skimage.transform import resize
from skimage.util import montage

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
from albumentations import Compose, HorizontalFlip
from albumentations.pytorch import ToTensorV2


def get_augmentations(phase):
    list_transforms = []

    list_trfms = Compose(list_transforms)
    return list_trfms


class BratsDataset(Dataset):
    def __init__(self, dir: str, phase: str = "test", is_resize: bool = False):

        self.dir = dir
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
        self.is_resize = is_resize

    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, idx):

        dir_name = os.listdir(self.dir)[idx]
        root_path = os.path.join(self.dir, dir_name)

        # load all modalities
        images = []
        for data_type in self.data_types:
            img_name = dir_name + data_type
            img_path = os.path.join(root_path, img_name)
            img = self.load_img(img_path)  # .transpose(2, 0, 1)

            if self.is_resize:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != "test":
            mask_name = dir_name + "_seg.nii.gz"
            mask_path = os.path.join(root_path, mask_name)
            mask = self.load_img(mask_path)

            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)

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

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (16, 16, 16), preserve_range=True)
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


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    dir: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """Returns: dataloader for the model training"""

    dataset = dataset(dir, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


if __name__ == "__main__":
    train_dir = "/home/sanchit/Segmentation Research/BraTS Data/loader_test"  # "../data/brats21/BraTS_2021_training"
    train_loader = get_dataloader(BratsDataset, train_dir, phase="train")
    data = next(iter(train_loader))
