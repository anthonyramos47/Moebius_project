# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def blur_shadow(img, blur=20, opacity=0.5):
    img[:, :, [0, 1, 2]] = 0.
    img[:, :, 3] = gaussian_filter(img[:, :, 3], sigma=blur) * opacity
    return img


def add_shadow(img, shadow):
    for i in range(3):
        img[:, :, i] = img[:, :, 3] * img[:, :, i] + (1 - img[:, :, 3]) * shadow[:, :, i]
    img[:, :, 3] = img[:, :, 3] + (1 - img[:, :, 3]) * shadow[:, :, 3]
    return img


def opacity_threshold(img, threshold=None):
    if threshold is None:
        return img
    img[:, :, 3] = np.where(img[:, :, 3] > threshold, 1., 0.)
    return img
