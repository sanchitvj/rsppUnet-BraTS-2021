import numpy as np
import scipy
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import affine_transform
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

import torch.nn as nn

# NOTE: random_shift, elastic_transfrom and random_rotate are taken from
# https://github.com/The-AI-Summer/learn-deep-learning/tree/main/Medical


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


def shift(img, mask, max_percentage=0.4):

    _, dim1, dim2, dim3 = img.shape
    m1, m2, m3 = (
        int(dim1 * max_percentage / 2),
        int(dim1 * max_percentage / 2),
        int(dim1 * max_percentage / 2),
    )
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)

    offset_matrix = np.array(
        [[0, 0, 0, 1], [1, 0, 0, d1], [0, 1, 0, d2], [0, 0, 1, d3]]
    )
    # print(offset_matrix.shape)
    img = ndimage.interpolation.affine_transform(img, offset_matrix)
    mask = ndimage.interpolation.affine_transform(mask, offset_matrix)

    return img, mask


def elastic_transform(image, mask, alpha=6, sigma=40, bg_val=0.1):
    """
    Elastic deformation of images as described in
    Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual
    Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Modified from:
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62

    Modified to take 3D inputs
    Deforms both the image and corresponding label file
    image linear/trilinear interpolated
    Label volumes nearest neighbour interpolated
    """
    assert image.ndim == 4  # we have 4 not 3
    im_shape = image.shape
    mk_shape = mask.shape
    # imgnew = np.squeeze(image, axis=0)
    # masknew = np.squeeze(mask, axis=0)

    dtype = image.dtype

    # Define coordinate system
    im_coords = (
        np.arange(im_shape[0]),
        np.arange(im_shape[1]),
        np.arange(im_shape[2]),
        np.arange(im_shape[3]),
    )
    mk_coords = (
        np.arange(mk_shape[0]),
        np.arange(mk_shape[1]),
        np.arange(mk_shape[2]),
        np.arange(mk_shape[3]),
    )

    # Get random elastic deformations
    # FIXME: im_shape may be an issue with masks.
    dx = (
        gaussian_filter(
            (np.random.rand(*im_shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (np.random.rand(*im_shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )
    dz = (
        gaussian_filter(
            (np.random.rand(*im_shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )

    mdx = (
        gaussian_filter(
            (np.random.rand(*mk_shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )
    mdy = (
        gaussian_filter(
            (np.random.rand(*mk_shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )
    mdz = (
        gaussian_filter(
            (np.random.rand(*mk_shape) * 2 - 1), sigma, mode="constant", cval=0.0
        )
        * alpha
    )
    # Define sample points
    ic, ix, iy, iz = np.mgrid[
        0 : im_shape[0], 0 : im_shape[1], 0 : im_shape[2], 0 : im_shape[3]
    ]
    mc, mx, my, mz = np.mgrid[
        0 : mk_shape[0], 0 : mk_shape[1], 0 : mk_shape[2], 0 : mk_shape[3]
    ]
    im_indices = (
        np.reshape(ic, (-1, 1)),
        np.reshape(ix + dx, (-1, 1)),
        np.reshape(iy + dy, (-1, 1)),
        np.reshape(iz + dz, (-1, 1)),
    )
    mk_indices = (
        np.reshape(mc, (-1, 1)),
        np.reshape(mx + mdx, (-1, 1)),
        np.reshape(my + mdy, (-1, 1)),
        np.reshape(mz + mdz, (-1, 1)),
    )

    # Initialize interpolators
    im_intrps = RegularGridInterpolator(
        im_coords, image, method="linear", bounds_error=False, fill_value=bg_val
    )
    mk_intrps = RegularGridInterpolator(
        mk_coords, mask, method="linear", bounds_error=False, fill_value=bg_val
    )

    # Interpolate 3D image image
    image = np.empty(shape=image.shape, dtype=dtype)
    image = im_intrps(im_indices).reshape(im_shape)

    # Interpolate labels
    # FIXME: ValueError: There are 4 points and 3 values in dimension 0...
    # ...not using image and labels together
    # if labels is not None:
    mask_new = np.empty(shape=mask.shape, dtype=dtype)
    mask_new = mk_intrps(mk_indices).reshape(mk_shape)

    # mk_intrp = RegularGridInterpolator(
    #     mk_coords, mask_new, method="nearest", bounds_error=False, fill_value=0
    # )

    # mask = mk_intrp(mk_indices).reshape(mk_shape).astype(mask.dtype)

    return image, mask_new


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

    if aug_name == "shift":
        imgnew, masknew = shift(img, masknew, args.max_percentage)

    if aug_name == "elastic_transform":
        imgnew, masknew = elastic_transform(imgnew, masknew)  # NOTE: no args for now

    return imgnew, masknew


def multi_augs(img, mask, args):
    """
    Randomly apply augmentations.
    """
    imgnew, masknew = img, mask

    aug_bool = np.random.choice([0, 1], size=4)
    if np.any(aug_bool[0]):
        imgnew, masknew = flip(imgnew, masknew)

    if np.any(aug_bool[1]):
        imgnew, masknew = brightness(imgnew, masknew)

    if np.any(aug_bool[2]):
        imgnew, masknew = rotate(imgnew, masknew, args.min_angle, args.max_angle)

    if np.any(aug_bool[3]):
        imgnew, masknew = shift(imgnew, masknew, args.max_percentage)

    #     if np.any(aug_bool[4]):
    #         imgnew, masknew = elastic_transform(imgnew, masknew)  # NOTE: no args for now

    return imgnew, masknew


def robust_augs(img, mask, args):
    """
    Apply all augmentations.
    """
    imgnew, masknew = img, mask

    if np.random.random_sample() < 0.7:
        imgnew, masknew = flip(imgnew, masknew)
    if np.random.random_sample() < 0.3:
        imgnew, masknew = brightness(imgnew, masknew)
    if np.random.random_sample() < 0.3:
        imgnew, masknew = rotate(imgnew, masknew, args.min_angle, args.max_angle)
    #     if np.random.random_sample() < 0.3:
    #         imgnew, masknew = shift(imgnew, masknew, args.max_percentage)
    #     imgnew, masknew = elastic_transform(imgnew, masknew)

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
