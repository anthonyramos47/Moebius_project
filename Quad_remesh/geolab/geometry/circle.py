#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'

# -----------------------------------------------------------------------------
#                                   Circle
# -----------------------------------------------------------------------------


def circle_three_points(p1, p2, p3):
    """ Makes a circle passing through three points.

    Parameters
    ----------
    p1 : np.array (n, 3)
        The first points.
    p2 : np.array (n, 3)
        The second points.
    p3 : np.array (n, 3)
        The third points.

    Returns
    -------
    frame : np.array (4, n, 3)
        The frames or circles. This is an array where 0 axis contains the
        centers, and 1, 2 , and 3 axes contains, respectively, the frames
        e1, e2, and e3.
    radius : np.array (n, )
        The radii of circles.

    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    t = p2 - p1
    u = p3 - p1
    v = p3 - p2
    n = np.cross(t, u)
    nsl = np.linalg.norm(n, axis=1) ** 2
    nsl[np.where(nsl == 0)] = 1e-20
    insl2 = 1.0 / (2.0 * nsl)
    insl3 = np.array([insl2, insl2, insl2]).T
    a = np.einsum('ij,ij->i', t, t) * np.einsum('ij,ij->i', u, v)
    b = np.einsum('ij,ij->i', t, v) * np.einsum('ij,ij->i', u, u)
    c = np.array([a, a, a]).T * u - np.array([b, b, b]).T * t
    C = p1 + insl3 * c
    r = (np.einsum('ij,ij->i', t, t) * np.einsum('ij,ij->i', u, u) *
         np.einsum('ij,ij->i', v, v) * insl2 * 0.5) ** 0.5
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    e2 = t / np.linalg.norm(t, axis=1, keepdims=True)
    e1 = np.cross(e2, n)
    frame = np.stack((C, e1, e2, n))
    return frame, r


def sample_circle(frame, radius, sampling=50):
    """ Compute sample points of a circle.

    Parameters
    ----------
    frame : np.array (4, n, 3)
        The frames of circles.
    radius : np.array (n,)
        The radii of circles.
    sampling : int
        The number of sampling points of each circle.

    Returns
    -------
    vertices : np.array (n * sampling, 3)
        The sampled points.

    """
    if type(radius) is float or type(radius) is int:
        radius = [radius]
    radius = np.array(radius)
    if len(radius) < len(frame[0]):
        radius = np.repeat(radius[0], len(frame[0]))
    N = frame.shape[1]
    phi = np.linspace(0, 2 * np.pi - 2 * np.pi / sampling, sampling)
    phi = np.tile(phi, N)
    r = np.repeat(radius, sampling)
    Ox = np.repeat(frame[0, :, 0], sampling)
    Oy = np.repeat(frame[0, :, 1], sampling)
    Oz = np.repeat(frame[0, :, 2], sampling)
    e1x = np.repeat(frame[1, :, 0], sampling)
    e1y = np.repeat(frame[1, :, 1], sampling)
    e1z = np.repeat(frame[1, :, 2], sampling)
    e2x = np.repeat(frame[2, :, 0], sampling)
    e2y = np.repeat(frame[2, :, 1], sampling)
    e2z = np.repeat(frame[2, :, 2], sampling)
    vx = Ox + r * np.sin(phi) * e1x + r * np.cos(phi) * e2x
    vy = Oy + r * np.sin(phi) * e1y + r * np.cos(phi) * e2y
    vz = Oz + r * np.sin(phi) * e1z + r * np.cos(phi) * e2z
    vertices = np.array([vx, vy, vz]).T
    return vertices
