import numpy as np
import scipy, time, elasticdeform
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import affine_transform
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

import torch.nn as nn

# NOTE: elastic_transfrom and random_rotate are taken from below
# https://github.com/The-AI-Summer/learn-deep-learning/tree/main/Medical


def exp_dim_mask(image, mask):

    mask_new = mask.copy()
    mask_new = np.pad(
        mask_new,
        ((0, 1), (0, 0), (0, 0), (0, 0)),
        mode="constant",
        constant_values=5,
    )
    # TODO: modify code structure
    image, mask = elastic_transform(image, mask_new)
    mask_new = np.delete(mask, 3, 0)

    return image, mask_new


def flip(img, mask):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
    # img = list(img)
    choice = np.random.randint(3)
    # INFO: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/10?u=sanchitvj
    if choice == 0:  # flip on x
        img_flip = img[:, ::-1, :, :] - np.zeros_like(img)
        mask_flip = mask[:, ::-1, :, :] - np.zeros_like(mask)
    if choice == 1:  # flip on y
        img_flip = img[:, :, ::-1, :] - np.zeros_like(img)
        mask_flip = mask[:, :, ::-1, :] - np.zeros_like(mask)
    if choice == 2:  # flip on z
        img_flip = img[:, :, :, ::-1] - np.zeros_like(img)
        mask_flip = mask[:, :, :, ::-1] - np.zeros_like(mask)
    return img_flip, mask_flip


def elastic_transform(image, mask, alpha=6, sigma=40, bg_val=0.1):

    image, mask_new = elasticdeform.deform_random_grid(
        [image, mask],
        sigma=2,
        axis=[(1, 2, 3), (1, 2, 3)],
        order=[1, 0],
        mode="constant",
    )

    return image, mask_new


def intensity_shift(img, mask, factor=0.1):

    scale_factor = np.random.uniform(
        1.0 - factor, 1.0 + factor, size=[1, img.shape[1], 1, img.shape[-1]]
    )
    shift_factor = np.random.uniform(
        -factor, factor, size=[1, img.shape[1], 1, img.shape[-1]]
    )

    image = img * scale_factor + shift_factor

    return image, mask


def rotate(img, mask, min_angle=-30, max_angle=30):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return:
    """
    assert img.ndim == 4, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    # all_axes = [(1, 0), (1, 2), (0, 2)]
    # NOTE: first is no. modality; each element increased by 1
    all_axes = [(2, 1), (2, 3), (1, 3)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]

    image = ndimage.rotate(img, angle, axes=axes, reshape=False)
    mask = ndimage.rotate(mask, angle, axes=axes, reshape=False)

    # FIXME: output looks like = ([1, 4, 32, 44, 44]), may be an issue later.
    return image, mask


def brightness(img, mask):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.

    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]

    new_im = gain * im^gamma
    """

    img_new = np.zeros(img.shape)
    for c in range(img.shape[-1]):
        im = img[:, :, :, c]
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(
            2,
        ) + 0.8
        im_new = np.sign(im) * gain * (np.abs(im) ** gamma)
        img_new[:, :, :, c] = im_new

    return img_new, mask


def test_aug(img, mask, aug_name, args):
    """
    Try a particular augmentation.
    """
    imgnew, masknew = img, mask

    if aug_name == "flip":
        imgnew, masknew = flip(imgnew, masknew)
    if aug_name == "brightness":
        imgnew, masknew = brightness(imgnew, masknew)
    if aug_name == "rotate":
        imgnew, masknew = rotate(imgnew, masknew, args.min_angle, args.max_angle)
    if aug_name == "elastic_transform":
        imgnew, masknew = exp_dim_mask(imgnew, masknew)  # NOTE: no args for now

    return imgnew, masknew


def multi_augs(img, mask, args):
    """
    Randomly apply augmentations.
    """
    imgnew, masknew = img, mask

    aug_bool = np.random.choice([0, 1], size=3)
    if np.any(aug_bool[0]):
        imgnew, masknew = flip(imgnew, masknew)
    if np.any(aug_bool[1]):
        imgnew, masknew = brightness(imgnew, masknew)
    if np.any(aug_bool[2]):
        imgnew, masknew = rotate(imgnew, masknew, args.min_angle, args.max_angle)
    if np.any(aug_bool[3]):
        imgnew, masknew = elastic_transform(imgnew, masknew)  # NOTE: no args for now

    return imgnew, masknew


def robust_augs(img, mask, args):
    """
    Apply all augmentations.
    """
    imgnew, masknew = img, mask

    if np.random.random_sample() < 0.5:
        imgnew, masknew = flip(imgnew, masknew)
    if np.random.random_sample() < 0.3:
        imgnew, masknew = brightness(imgnew, masknew)
    if np.random.random_sample() < 0.3:
        imgnew, masknew = rotate(imgnew, masknew, args.min_angle, args.max_angle)
    if np.random.random_sample() < 0.3:
        imgnew, masknew = exp_dim_mask(imgnew, masknew)
    if np.random.random_sample() < 0.3:
        imgnew, masknew = intensity_shift(imgnew, masknew)

    return imgnew, masknew


class DataAugmentation(nn.Module):
    def __init__(self, args, p=0.75):
        super(DataAugmentation, self).__init__()

        self.args = args
        self.aug_name = args.aug_name
        self.p = p
        self.aug_type = args.aug_type

    def forward(self, x, y):

        # Apply augs to 75% samples.
        if np.random.random_sample() < self.p:

            if self.aug_name is not None:
                x, y = test_aug(x, y, self.aug_name, self.args)

            elif self.aug_type == "multiple":
                x, y = multi_augs(x, y, self.args)

            elif self.aug_type == "robust":
                x, y = robust_augs(x, y, self.args)

            return x, y
        else:
            return x, y
