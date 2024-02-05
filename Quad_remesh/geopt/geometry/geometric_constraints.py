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


# -----------------------------------------------------------------------------
#                                  FIXED
# -----------------------------------------------------------------------------

def vector_length_constraints(unknowns_vector=None, number_of_vectors=None,
                              length=1, vectors_offset=0, weight=1, **kwargs):
    w = weight / length
    X = unknowns_vector
    V = number_of_vectors
    v = np.arange(V)
    i = np.hstack((v, v, v))
    j = np.hstack((v, v + V, v + 2*V)) + vectors_offset
    data = 2 * np.hstack((X[v], X[v + V], X[v + 2*V])) * w
    r = (X[v]**2 + X[v + V]**2 + X[v + 2*V]**2 + length) * weight
    H = sparse.coo_matrix((data, (i, j)), shape=(V, len(unknowns_vector)))
    return H, r
