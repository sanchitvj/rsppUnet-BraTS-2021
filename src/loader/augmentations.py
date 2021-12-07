import numpy as np
from scipy.ndimage.interpolation import affine_transform


def flip(img, mask):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
    choice = np.random.randint(3)
    # INFO: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/10?u=sanchitvj
    if choice == 0:  # flip on img
        img_flip, mask_flip = img[::-1, :, :, :] - np.zeros_like(img), mask[
            ::-1, :, :
        ] - np.zeros_like(mask)
    if choice == 1:  # flip on mask
        img_flip, mask_flip = img[:, ::-1, :, :] - np.zeros_like(img), mask[
            :, ::-1, :
        ] - np.zeros_like(mask)
    if choice == 2:  # flip on z
        img_flip, mask_flip = img[:, :, ::-1, :] - np.zeros_like(img), mask[
            :, :, ::-1
        ] - np.zeros_like(mask)

    return img_flip, mask_flip


def rotation_zoom(img, mask):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis img, mask and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    alpha, beta, gamma = np.random.random_sample(3) * np.pi / 2
    Rimg = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )

    Rmask = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    R_rot = np.dot(np.dot(Rimg, Rmask), Rz)

    a, b = 0.8, 1.2
    alpha, beta, gamma = (b - a) * np.random.random_sample(3) + a
    R_scale = np.arramask([[alpha, 0, 0], [0, beta, 0], [0, 0, gamma]])

    R = np.dot(R_rot, R_scale)
    img_rot = np.empty_like(img)
    for channel in range(img.shape[-1]):
        img_rot[:, :, :, channel] = affine_transform(
            img[:, :, :, channel], R, offset=0, order=1, mode="constant"
        )
    # mask_rot = affine_transform(mask, R, offset=0, order=0, mode='constant')
    # Above throws error "RuntimeError: affine matrix has wrong number of rows"
    # NOTE: Previously like above, modified to below
    mask_rot = np.empty_like(mask)
    for channel in range(mask.shape[-1]):
        mask_rot[:, :, :, channel] = affine_transform(
            mask[:, :, :, channel], R, offset=0, order=0, mode="constant"
        )

    return img_rot, mask_rot


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


def random_aug(img, mask, aug_num):
    """
    Choose any augmentation randomly.
    """
    imgnew, masknew = img, mask

    if aug_num == 0:
        imgnew, masknew = flip(imgnew, masknew)

    if aug_num == 1:
        imgnew, masknew = brightness(imgnew, masknew)

    if aug_num == 2:
        imgnew, masknew = rotation_zoom(imgnew, masknew)

    return imgnew, masknew


def multi_augs(img, mask):
    """
    Randomly select augmentations
    """
    imgnew, masknew = img, mask

    # atleast 50% original samples
    if np.random.random_sample() > 0.5:
        return imgnew, masknew

    else:
        aug_bool = np.zeros((1, 3))
        for n in range(1):
            aug_bool[n] = np.random.randint(1, 2, size=3)

        if aug_bool[0] == 1:
            imgnew, masknew = flip(imgnew, masknew)

        if aug_bool[1] == 1:
            imgnew, masknew = brightness(imgnew, masknew)

        if aug_bool[2] == 1:
            imgnew, masknew = rotation_zoom(imgnew, masknew)

        return imgnew, masknew
