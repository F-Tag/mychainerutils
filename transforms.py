import numpy as np


def cutout(image_origin, mask_ratio=0.33, rate=0.5):
    """
    image_origin CHW... array

    reference
    https://arxiv.org/abs/1708.04552
    https://www.kumilog.net/entry/numpy-data-augmentation
    """

    image = np.copy(image_origin)
    if np.random.rand() < rate:
        mask_value = image.mean()

        h, w = image.shape[1:3]
        mask_h = int(h * mask_ratio)
        mask_w = int(w * mask_ratio)
        top = np.random.randint(0 - mask_h // 2, h - mask_h)
        left = np.random.randint(0 - mask_w // 2, w - mask_w)
        bottom = top + mask_h
        right = left + mask_w

        if top < 0:
            top = 0
        if left < 0:
            left = 0

        image[:, top:bottom, left:right].fill(mask_value)

    return image


def rgb2gray(image, rate=0.5):

    if np.random.rand() < rate:
        gray = (0.2989 * image[0] + 0.5870 * image[1] +
                0.1140 * image[2]).astype(image.dtype)

        image = np.broadcast_to(gray, image.shape)

    return image
