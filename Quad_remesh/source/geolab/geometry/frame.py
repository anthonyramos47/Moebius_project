#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

# -----------------------------------------------------------------------------

from geolab.utilities.arrayutilities import format_array, bounding_shape, \
    orthogonal_vectors, normalize

# -----------------------------------------------------------------------------

'''frame generator'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
#                                   Frame
# -----------------------------------------------------------------------------


def make_frame(origin=None, e1=None, e2=None, e3=None):
    """ Makes an array representing a set of frames. All input arrays are
    extended to match the shape of the longest one.

    Parameters
    ----------
    origin : np.array
        The array containing the coordinates of frame centers. If 'None',
        (0,0,0) is taken.
    e1 : np.array
        The array of e1 axes. If 'None', (1,0,0) is taken.
    e2 : np.array
        The array of e3 axes. If 'None', (0,1,0) is taken.
    e3 : np.array
        The array of e3 axes. If 'None', (0,0,1) is taken.

    Returns
    -------
    frame : np.array
        The frame. This is an array where 0 axis contains the centers, and
        1, 2 , and 3 axes contains, respectively, the frames e1, e2, and e3.

    """
    if origin is not None:
        if e1 is None and e2 is None and e3 is None:
            e3 = np.zeros(origin.shape)
            e3[:, 2] = 1
    if origin is None:
        origin = [[0, 0, 0]]
    if e1 is None and e2 is None:
        if e3 is None:
            e1 = [[1, 0, 0]]
            e2 = [[0, 1, 0]]
            e3 = [[0, 0, 1]]
        else:
            e3 = normalize(e3)
            e1 = orthogonal_vectors(e3)
            e2 = np.cross(e3, e1)
    if e1 is None and e2 is not None:
        if e3 is None:
            e3 = [[0, 0, 1]]
        else:
            e3 = normalize(e3)
        e2 = normalize(e2)
        e1 = np.cross(e2, e3)
    if e1 is not None and e2 is None:
        if e3 is None:
            e3 = [[0, 0, 1]]
        else:
            e3 = normalize(e3)
        e1 = normalize(e1)
        e2 = np.cross(e3, e1)
    shape = bounding_shape([origin, e1, e2, e3])
    if len(shape) == 1:
        shape = [None, shape[0]]
    origin = format_array(origin, shape=shape)
    e1 = format_array(e1, shape=shape)
    e2 = format_array(e2, shape=shape)
    e3 = format_array(e3, shape=shape)
    frame = np.stack((origin, e1, e2, e3))
    return frame


def rotate_frame(frame, theta=np.pi / 2, axis=3):
    rot_frame = np.copy(frame).astype(float)
    rot_axes = [1, 2, 3]
    rot_axes.remove(axis)
    sin = np.sin(theta)
    cos = np.cos(theta)
    R = np.zeros((len(frame[0]), 3, 3), dtype=float)
    wx = frame[axis, :, 0]
    wy = frame[axis, :, 1]
    wz = frame[axis, :, 2]
    R[:, 0, 0] = cos + wx ** 2 * (1 - cos)
    R[:, 0, 1] = wx * wy * (1 - cos) - wz * sin
    R[:, 0, 2] = wy * sin + wx * wz * (1 - cos)
    R[:, 1, 0] = wz * sin + wx * wy * (1 - cos)
    R[:, 1, 1] = cos + wy ** 2 * (1 - cos)
    R[:, 1, 2] = -wx * sin + wy * wz * (1 - cos)
    R[:, 2, 0] = -wy * sin + wx * wz * (1 - cos)
    R[:, 2, 1] = wx * sin + wy * wz * (1 - cos)
    R[:, 2, 2] = cos + wz ** 2 * (1 - cos)
    rot_frame[rot_axes[0]] = np.einsum('ijk,ik->ij', R, frame[rot_axes[0]])
    rot_frame[rot_axes[1]] = np.einsum('ijk,ik->ij', R, frame[rot_axes[1]])
    return rot_frame


if __name__ == '__main__':
    from geolab.plot.viewer import plotter

    f = make_frame(origin=np.array([[1, 0, 0], [0, 1, 1]]),
                   e3=np.array([[1, 0, 0], [0, 1, 1]]))
    r = rotate_frame(f, theta=[0.1, 1], axis=2)
    p = plotter()
    p.plot_frame(f)
    p.plot_frame(r)
    p.show()
