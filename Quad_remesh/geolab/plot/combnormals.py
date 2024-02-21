#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

# -----------------------------------------------------------------------------

from geolab.utilities import arrayutilities

from geolab.utilities.guidedprojection import GuidedProjection

from geolab.geometry.polyline import polyline_vertex_frame

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class CombNormals(GuidedProjection):
    _polyline = None

    _untwist = False

    def __init__(self, vertices, closed=False):
        GuidedProjection.__init__(self)

        self.vertices = vertices

        self.closed = closed

        self.verbose = False

        weights = {

            'geometric': 1,

            'normals': 10,

        }

        self.add_weights(weights)

        self.initialization()

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------

    @property
    def polyline(self):
        return self._polyline

    @polyline.setter
    def polyline(self, polyline):
        self._polyline = polyline
        self.initialization()

    # --------------------------------------------------------------------------
    #                               Initialization
    # --------------------------------------------------------------------------

    def on_initialize(self):
        T, N, B = polyline_vertex_frame(self.vertices, self.closed)
        N = self.align_normals(N)
        self.add_value('tangents', T)
        self.add_value('normals', N)

    def set_dimensions(self):
        self._N = 3 * len(self.vertices)

    def initialize_unknowns_vector(self):
        normals = self.get_value('normals')
        X = normals.flatten('F')
        self._X = X
        self._X0 = np.copy(X)

    def make_errors(self):
        pass

    def post_iteration_update(self):
        pass

    def on_reinitilize(self):
        pass

    # -------------------------------------------------------------------------
    #                                   Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def align_normals(N):
        dot = np.einsum('ij,ij->i', N[:-1], N[1:])
        flip = np.where(dot < 0)[0] + 1
        for i in flip:
            N[i:] *= -1
        return N

    def normals(self):
        return np.reshape(self.X, (-1, 3), order='F')

    def plot_normals(self):
        from geolab.plot.viewer import plotter
        plotter = plotter()
        plotter.plot_vectors(self.normals(),
                             anchor=self.vertices,
                             position='tail')
        plotter.plot_vectors(self.get_value('normals'),
                             anchor=self.vertices, color='b', position='tail')
        plotter.plot_edges(self.vertices, closed=self.closed)
        plotter.show()

    # -------------------------------------------------------------------------
    #                          Geometric Constraints
    # -------------------------------------------------------------------------

    def unit_constraints(self):
        w = 10
        V = len(self.vertices)
        v = np.arange(V)
        i = np.arange(V)
        i = np.hstack((i, i, i))
        j = np.hstack((v, V + v, 2 * V + v))
        X = self.X
        data = 2 * np.hstack((X[v], X[V + v], X[2 * V + v])) * w
        H = sparse.coo_matrix((data, (i, j)), shape=(V, self.N))
        r = ((X[v] ** 2 + X[V + v] ** 2 + X[2 * V + v] ** 2) + 1) * w
        self.add_iterative_constraint(H, r, 'unit')

    def tangent_constraints(self):
        w = 10  # - self.get_value('local_w')
        V = len(self.vertices)
        t = self.get_value('tangents')
        v = np.arange(V)
        i = np.arange(V)
        i = np.hstack((i, i, i))
        j = np.hstack((v, V + v, 2 * V + v))
        data = np.hstack((t[:, 0] * w, t[:, 1] * w, t[:, 2] * w))
        H = sparse.coo_matrix((data, (i, j)), shape=(V, self.N))
        r = np.zeros(V)
        self.add_constant_constraint(H, r, 'tangent')

    def normals_fairness(self):
        w = 0.1 / len(self.vertices)
        v0 = np.arange(len(self.vertices))
        v2 = np.roll(v0, 1)
        v1 = np.roll(v0, -1)
        if not self.closed:
            v0 = v0[1:-2]
            v2 = v2[1:-2]
            v1 = v1[1:-2]
        W = len(v0)
        V = len(self.vertices)
        one = np.ones(W)
        i = np.arange(W)
        i = np.hstack((i, i, i))
        j = np.hstack((v0, v1, v2))
        data = np.hstack((-2 * one, one, one)) * w
        i = np.hstack((i, W + i, 2 * W + i))
        j = np.hstack((j, V + j, 2 * V + j))
        data = np.hstack((data, data, data))
        K = sparse.coo_matrix((data, (i, j)), shape=(3 * W, self.N))
        s = np.zeros(3 * W)
        self.add_constant_fairness(K, s, 'fairness')

    def difference_fairness(self):
        w = min(10 / len(self.vertices), 0.1)
        v0 = np.arange(len(self.vertices))
        v1 = np.roll(v0, 1)
        if not self.closed:
            v0 = v0[1:]
            v1 = v1[1:]
        W = len(v0)
        V = len(self.vertices)
        one = np.ones(W)
        i = np.arange(W)
        i = np.hstack((i, i))
        j = np.hstack((v0, v1))
        data = np.hstack((-one, one)) * w
        i = np.hstack((i, W + i, 2 * W + i))
        j = np.hstack((j, V + j, 2 * V + j))
        data = np.hstack((data, data, data))
        K = sparse.coo_matrix((data, (i, j)), shape=(3 * W, self.N))
        s = np.zeros(3 * W)
        self.add_constant_fairness(K, s, 'fairness')

    # -------------------------------------------------------------------------
    #                                 Build
    # -------------------------------------------------------------------------

    def build_iterative_constraints(self):
        self.unit_constraints()

    def build_constant_constraints(self):
        self.tangent_constraints()

    def build_constant_fairness(self):
        self.difference_fairness()
        self.normals_fairness()


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

def comb_normals(vertices, closed=False):
    optimizer = CombNormals(vertices, closed)
    optimizer.threshold = 1e-5
    optimizer.iterations = 10
    optimizer.epsilon = 0.0001
    optimizer.optimize()
    #optimizer.plot_normals()
    return optimizer.normals()
