#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import copy

import numpy as np

from scipy import sparse

# -----------------------------------------------------------------------------

import geolab as geo

# -----------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'

"""
vertices
halfedges
unknowns_vector
"""


# -----------------------------------------------------------------------------
#                                  FIXED
# -----------------------------------------------------------------------------

def fixed_vertices_constraints(vertices=None, fixed_vertices=None,
                               vertices_offset=0, unknowns_vector=None, weight=1,
                               **kwargs):
    d = np.max(np.abs(vertices))
    w = weight / d
    V = len(vertices)
    W = len(fixed_vertices)
    v = np.array(fixed_vertices)
    j = np.hstack((v, V + v, 2 * V + v)) + vertices_offset
    i = np.arange(W)
    i = np.hstack((i, W + i, 2 * W + i))
    r = vertices[np.array(fixed_vertices), :]
    r = np.reshape(r, 3 * W, order='F') * w
    data = np.ones([3 * W]) * w
    H = sparse.coo_matrix((data, (i, j)), shape=(3 * W, len(unknowns_vector)))
    return H, r


# -----------------------------------------------------------------------------
#                                  GLIDING
# -----------------------------------------------------------------------------

def reference_gliding_constraints(vertices=None, reference_vertices=None,
                                  reference_halfedges=None,
                                  vertices_offset=0, unknowns_vector=None, weight=1,
                                  **kwargs):
    d = np.max(np.abs(vertices))
    w = weight / d
    V = len(vertices)
    normals = geo.vertices_normal(reference_vertices, reference_halfedges)
    closest = geo.closest_vertices(reference_vertices, vertices)
    points = reference_vertices[closest, :]
    normals = normals[closest, :]
    r = w * np.einsum('ij,ij->i', points, normals)
    data = w * normals.flatten('F')
    v = np.arange(V)
    j = np.hstack((v, V + v, 2 * V + v)) + vertices_offset
    i = np.hstack((v, v, v))
    H = sparse.coo_matrix((data, (i, j)), shape=(V, len(unknowns_vector)))
    return H, r


# -----------------------------------------------------------------------------
#                                  GLIDING
# -----------------------------------------------------------------------------

def edge_lengths_auxiliary_constraints(vertices=None, halfedges=None,
                                       vertices_offset=0, lengths_offset=0,
                                       unknowns_vector=None, weight=1, **kwargs):
    V = len(vertices)
    E = geo.number_of_edges(halfedges)
    N = len(unknowns_vector)
    X = unknowns_vector
    i = np.arange(E)
    e = i + lengths_offset
    v = geo.edges(halfedges)
    v1 = v[:, 0] + vertices_offset
    v2 = v[:, 1] + vertices_offset
    r = (X[v1] ** 2 + X[v2] ** 2 - 2 * X[v1] * X[v2]
         + X[V + v1] ** 2 + X[V + v2] ** 2 - 2 * X[V + v1] * X[V + v2]
         + X[2 * V + v1] ** 2 + X[2 * V + v2] ** 2 - 2 * X[2 * V + v1] * X[2 * V + v2]
         - X[e] ** 2) * weight
    v1 = np.hstack((v1, V + v1, 2 * V + v1))
    v2 = np.hstack((v2, V + v2, 2 * V + v2))
    i = np.hstack((i, i, i, i, i, i, i))
    j = np.hstack((v1, v2, e))
    data = 2 * np.hstack((X[v1] - X[v2], X[v2] - X[v1], -X[e])) * weight
    H = sparse.coo_matrix((data, (i, j)), shape=(E, N))
    return H, r


# -----------------------------------------------------------------------------
#                                  FAIRNESS
# -----------------------------------------------------------------------------

def laplacian_fairness_constraints(vertices=None, halfedges=None, vertices_offset=0,
                                   unknowns_vector=None, weight=1, **kwargs):
    d = np.max(np.abs(vertices))
    w = weight / d
    V = len(vertices)
    N = len(unknowns_vector)
    bound = geo.boundary_vertices(halfedges)
    v0, vj, l = geo.vertices_ring_vertices(halfedges, order=True, return_lengths=True)
    inner = np.invert(np.in1d(v0, bound))
    v0 = v0[inner]
    vj = vj[inner]
    quad = np.in1d(v0, np.where(l == 4)[0])
    other = np.invert(quad)
    v0o = v0[other]
    if v0o.shape[0] > 0:
        i1 = geo.repeated_range(v0o)
        j1 = vj[other]
        data1 = -np.ones(v0o.shape[0]) / l[v0o]
        off1 = np.amax(i1) + 1
        i2 = np.arange(off1)
        j2 = np.unique(v0o)
        data2 = np.ones(j2.shape[0])
    else:
        i1 = i2 = j1 = j2 = data1 = data2 = np.array([])
        off1 = 0
    v0q = v0[quad][0::2]
    if v0q.shape[0] > 0:
        i3 = geo.repeated_range(v0q, offset=off1)
        j3 = vj[quad][0::2]
        data3 = np.repeat(-.5, v0q.shape[0])
        i4 = i3 + np.amax(i3) + 1 - off1
        j4 = vj[quad][1::2]
        data4 = data3
        i5 = np.arange(off1, off1 + v0q.shape[0])
        j5 = np.unique(v0q)
        data5 = np.ones(i5.shape[0])
    else:
        i3 = i4 = i5 = j3 = j4 = j5 = data3 = data4 = data5 = np.array([])
    i = np.hstack((i1, i2, i3, i4, i5))
    j = np.hstack((j1, j2, j3, j4, j5, j5)) + vertices_offset
    data = w * np.hstack((data1, data2, data3, data4, data5))
    W = int(np.amax(i)) + 1
    Kx = sparse.coo_matrix((data, (i, j)), shape=(W, N))
    j = V + j
    Ky = sparse.coo_matrix((data, (i, j)), shape=(W, N))
    j = V + j
    Kz = sparse.coo_matrix((data, (i, j)), shape=(W, N))
    K = sparse.vstack([Kx, Ky, Kz])
    s = np.zeros(int(3 * (W + 1)))
    return K, s




