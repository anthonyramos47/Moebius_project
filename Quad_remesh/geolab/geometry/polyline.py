#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

from scipy import spatial

from scipy.sparse import linalg

# -----------------------------------------------------------------------------

from geolab.utilities.arrayutilities import normalize

from geolab.utilities.stringutilities import make_filepath

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
#                                   POLYLINE
# -----------------------------------------------------------------------------


def refine_polyline(vertices, closed=False, steps=5, corner_tolerance=None):
    if steps <= 0:
        return
    V = len(vertices)
    corners = polyline_corners(vertices, closed, corner_tolerance)
    if closed:
        N = steps * V
    else:
        N = steps * (V - 1)
    v = np.arange(V)
    s = np.arange(V + N)
    d = np.repeat(3, V + N)
    d1 = np.repeat(-2, V + N)
    d2 = np.repeat(0.5, V + N)
    d[(steps + 1) * v] = 1
    d1[(steps + 1) * v] = 0
    d2[(steps + 1) * v] = 0
    i = np.hstack((s, s, s, s, s))
    j = np.hstack((s, np.roll(s, 1), np.roll(s, -1),
                   np.roll(s, 2), np.roll(s, -2)))
    data = np.hstack((d, d1, d1, d2, d2))
    if not closed:
        data[1] = data[V + N - 2] = 2.5
        data[V + N + 1] = data[3 * (V + N) - 2] = -1
        data[3 * (V + N) + 1] = 0
        data[5 * (V + N) - 2] = 0
        if corners is not None and len(corners) > 0:
            corners = np.array(corners)
            corners = np.delete(corners, np.where(corners == 0)[0])
    if corners is not None and len(corners) > 0:
        c = (steps + 1) * np.array(corners)
        data[np.roll(s, 1)[c]] = 2.5
        data[np.roll(s, -1)[c]] = 2.5
        data[V + N + np.roll(s, -1)[c]] = -1
        data[2 * (V + N) + np.roll(s, 1)[c]] = -1
        data[3 * (V + N) + np.roll(s, -1)[c]] = 0
        data[4 * (V + N) + np.roll(s, 1)[c]] = 0
    M = sparse.coo_matrix((data, (i, j)), shape=(V + N, V + N))
    P = np.zeros((N + V, 3))
    P[(steps + 1) * v] = vertices
    M = sparse.csc_matrix(M)
    X = linalg.spsolve(M, P[:, 0], use_umfpack=False)
    Y = linalg.spsolve(M, P[:, 1], use_umfpack=False)
    Z = linalg.spsolve(M, P[:, 2], use_umfpack=False)
    P = np.vstack((X, Y, Z)).T
    return P


def read_polyline(file_name):
    file_name = str(file_name)
    obj_file = open(file_name, encoding='utf-8')
    polylines = []
    vertices_list = []
    for l in obj_file:
        split_line = l.split(' ')
        if split_line[0] == 'v':
            split_x = split_line[1].split('\n')
            x = float(split_x[0])
            split_y = split_line[2].split('\n')
            y = float(split_y[0])
            split_z = split_line[3].split('\n')
            try:
                z = float(split_z[0])
            except ValueError:
                print('WARNING: disable line wrap when saving .obj')
            vertices_list.append([x, y, z])
        if split_line[0] == 'end\n':
            polylines.append(np.array(vertices_list))
            vertices_list = []
    if len(polylines) == 0:
        return np.array(vertices_list)
    if len(polylines) == 1:
        return polylines[0]
    return polylines


def save_polyline(vertices, file_name='polyline', overwrite=False, closed=False):
    """Save the polyline as OBJ file.

    Parameters
    ----------
    vertices : np.array (H, 3)
        The array of vertices.

    Optional Parameters
    -------------------
    file_name : str
        The path of the OBJ file to be created, without extension.
    overwrite : bool
        If `False` (default), when the path already exists, a sequential
        number is added to file_name. If `True`, existing paths are
        overwritten.
    closed : bool

    Returns
    -------
    str
        The path of the saves OBJ file (without extension).
    """
    path = make_filepath(file_name, 'obj', overwrite)
    obj = open(path, 'w')
    line = 'o {}\n'.format(file_name)
    obj.write(line)
    for v in range(len(vertices)):
        vi = vertices[v]
        line = 'v {} {} {}\n'.format(vi[0], vi[1], vi[2])
        obj.write(line)
    obj.write('l ')
    for v in range(len(vertices)):
        vf = str(v + 1)
        obj.write(vf + ' ')
        if closed:
            obj.write('1 ')
    obj.write('\n')
    obj.close()
    out = 'OBJ saved in {}'.format(path)
    print(out)
    return path.split('.')[0]


def polyline_edges_length(vertices, closed=False):
    next_vertex = np.roll(vertices, -1, axis=0)
    L = np.linalg.norm(next_vertex - vertices, axis=1)
    if not closed:
        L = L[:-1]
    return L


def polyline_vertex_bisectors(vertices, closed=False, return_cosines=False):
    """

    Parameters
    ----------
    vertices : np.array (n, 3)
    closed : Bool
    return_cosines : Bool

    Returns
    -------
    bisectors : np.array (n, 3)
    cosines : np.array (n,) (optional)
    """
    v0 = np.arange(len(vertices))
    v2 = np.roll(v0, 1)
    v1 = np.roll(v0, -1)
    V1 = vertices[v1, :] - vertices[v0, :]
    V2 = vertices[v0, :] - vertices[v2, :]
    V1 = normalize(V1)
    V2 = normalize(V2)
    B = V1 - V2
    B = normalize(B)
    if not closed:
        N0 = np.cross(np.cross(V1[0], B[1, :]), V1[0])
        B[0, :] = N0 / np.linalg.norm(N0)
        N1 = np.cross(np.cross(V2[-1], B[-2, :]), V2[-1])
        B[-1, :] = N1 / np.linalg.norm(N1)
    if not return_cosines:
        return B
    else:
        A = np.einsum('ij,ij->i', V1, -V2)
        if not closed:
            A[0] = A[-1] = -1
        return B, A


def polyline_vertex_frame(vertices, closed=False, return_cosine=False):
    v0 = np.arange(len(vertices))
    v2 = np.roll(v0, 1)
    v1 = np.roll(v0, -1)
    V1 = vertices[v1, :] - vertices[v0, :]
    V2 = vertices[v0, :] - vertices[v2, :]
    V1 = normalize(V1)
    V2 = normalize(V2)
    N = normalize(V1 - V2)
    T = normalize(V1 + V2)
    B = normalize(np.cross(V1, V2))
    k = np.einsum('ij,ij->i', V1, V2)
    if not closed:
        T[0] = V1[0]
        T[-1] = V2[-1]
        B[0] = B[1]
        B[-1] = B[-2]
        N[0] = np.cross(T[0], B[0])
        N[-1] = np.cross(T[-1], B[-1])
        k[0] = k[-1] = 1
    if return_cosine:
        return T, N, B, k
    else:
        return T, N, B


def polyline_corners(vertices, closed=False, corner_tolerance=0.5):
    if corner_tolerance is None:
        return []
    v1 = np.arange(len(vertices))
    v0 = np.roll(v1, 1)
    v2 = np.roll(v1, -1)
    if not closed:
        v0[0] = len(vertices) - 1
        v2[-1] = 0
    V1 = vertices[v2, :] - vertices[v1, :]
    V1 = normalize(V1)
    V2 = vertices[v1, :] - vertices[v0, :]
    V2 = normalize(V2)
    C = np.einsum('ij,ij->i', V1, V2)
    corners = v1[np.where(C[:] < corner_tolerance)[0]].tolist()
    return corners


def polyline_bishop_frame(vertices, closed=False):
    nV = len(vertices)
    T, N, B, c = polyline_vertex_frame(vertices, closed, return_cosine=True)
    s = (1 - c**2)**.5
    R_11 = c + B[:, 0]**2 * (1 - c)
    R_12 = B[:, 0] * B[:, 1] * (1 - c) - B[:, 2] * s
    R_13 = B[:, 0] * B[:, 2] * (1 - c) + B[:, 1] * s
    R_21 = B[:, 1] * B[:, 0] * (1 - c) + B[:, 2] * s
    R_22 = c + B[:, 1]**2 * (1 - c)
    R_23 = B[:, 1] * B[:, 2] * (1 - c) - B[:, 0] * s
    R_31 = B[:, 2] * B[:, 0] * (1 - c) - B[:, 1] * s
    R_32 = B[:, 2] * B[:, 1] * (1 - c) + B[:, 0] * s
    R_33 = c + B[:, 2]**2 * (1 - c)
    v = np.arange(nV)
    i = np.hstack((v, v, v, nV+v, nV+v, nV+v, 2*nV+v, 2*nV+v, 2*nV+v))
    j = np.hstack((v, nV+v, 2*nV+v, v, nV+v, 2*nV+v, v, nV+v, 2*nV+v))
    i = np.hstack((i, v, nV+v, 2*nV+v))
    roll = np.roll(v, 1)
    ones = np.ones(nV)
    ones[0] = 0
    R_11[0] = R_22[0] = R_33[0] = 1
    R_12[0] = R_12[0] = R_21[0] = R_23[0] = R_31[0] = R_32[0] = 0
    j = np.hstack((j, roll, nV + roll, 2*nV + roll))
    data = np.hstack((R_11, R_12, R_13, R_21, R_22, R_23, R_31, R_32, R_33,
                      -ones, -ones, -ones))
    b = np.zeros(3 * nV)
    b[0] = N[0, 0]
    b[nV] = N[0, 1]
    b[2*nV] = N[0, 2]
    A = sparse.coo_matrix((data, (i, j)), shape=(3*nV, 3*nV))
    E1 = sparse.linalg.spsolve(A.tocsc(), b)
    return np.reshape(E1, (nV, 3), order='F')


def polyline_pipe_frame(vertices, closed=False):
    return polyline_vertex_frame(vertices, closed, return_cosine=True)

