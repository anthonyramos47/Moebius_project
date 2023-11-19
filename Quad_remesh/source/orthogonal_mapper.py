# -*- coding: utf-8 -*-


# !/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

import scipy

# -----------------------------------------------------------------------------

import sys

import geopt

import geolab as geo

# -----------------------------------------------------------------------------

Base = geopt.GuidedProjection


class OrthoOptimizer(Base):
    _vertices = None

    _faces = None

    _halfedges = None

    _v1 = None

    _v2 = None

    _x1 = None

    _x2 = None

    def __init__(self, vertices=None, halfedges=None, v1=None, v2=None):
        Base.__init__(self)

        weights = {

            'orthogonal': 1,

            'closeness': 0.001,

            'mesh_fairness': 0.1

        }

        self.add_weights(weights)

        self._vertices = vertices

        self._halfedges = halfedges

        self._faces = np.array(geo.faces_list(halfedges))

        self._v1 = v1

        self._v2 = v2

        self._make_local_coordinates()

        self.make_residual = True

    # -------------------------------------------------------------------------
    # STANDARD
    # -------------------------------------------------------------------------

    def set_dimensions(self):
        self._N = 3 * len(self._vertices)

    def initialize_unknowns_vector(self):
        Base.initialize_unknowns_vector(self)
        X = self._vertices.flatten('F')
        self._X = X
        self._X0 = np.copy(X)

    def make_errors(self):
        pass

    def post_iteration_update(self):
        pass

    # -------------------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------------------

    def optimized_vertices(self):
        V = len(self._vertices)
        return np.reshape(self.X, (V, 3), order='F')

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def _make_local_coordinates(self):
        e1 = self._vertices[self._faces[:, 1]] - self._vertices[self._faces[:, 0]]
        e2 = self._vertices[self._faces[:, 2]] - self._vertices[self._faces[:, 0]]
        e3 = np.cross(e1, e2)
        J = np.array([e1.T, e2.T, e3.T]).T
        x1 = np.linalg.solve(J, self._v1)
        self._x1 = x1[:, [0, 1]]
        x2 = np.linalg.solve(J, self._v2)
        self._x2 = x2[:, [0, 1]]

    def vectors(self):
        V = self.optimized_vertices()
        e1 = V[self._faces[:, 1]] - V[self._faces[:, 0]]
        e2 = V[self._faces[:, 2]] - V[self._faces[:, 0]]
        V1 = np.einsum('i,ij->ij', self._x1[:, 0], e1) + np.einsum('i,ij->ij', self._x1[:, 1], e2)
        V2 = np.einsum('i,ij->ij', self._x2[:, 0], e1) + np.einsum('i,ij->ij', self._x2[:, 1], e2)
        n1 = np.linalg.norm(V1, axis=1, keepdims=True)
        n2 = np.linalg.norm(V2, axis=1, keepdims=True)
        n1[n1 == 0] = 1
        n2[n2 == 0] = 1
        V1 = V1 / n1
        V2 = V2 / n2
        return V1, V2

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    def save_directions_obj(self, file_name, scale=1, V1=None, V2=None, save_all=False):
        F = len(self._faces)
        B = geo.faces_centroid(self.optimized_vertices(), self._halfedges)
        mean = geo.edges_length(self.optimized_vertices(), self._halfedges)
        mean = np.min(mean)/2
        if V1 is None:
            V1, V2 = self.vectors()
        P1a = B - scale * mean * V1
        P1b = B + scale * mean * V1
        P2a = B - scale * mean * V2
        P2b = B + scale * mean * V2
        P = np.vstack((P1a, P1b, P2a, P2b))
        name = '{}_crossfield.obj'.format(file_name)
        obj = open(name, 'w')
        line = 'o {}\n'.format('lines')
        obj.write(line)
        for f in range(4 * F):
            vi = P[f]
            line = 'v {} {} {}\n'.format(vi[0], vi[1], vi[2])
            obj.write(line)
        if save_all:
            for f in range(F):
                line = 'l {} {}\n'.format(f + 1, f + F + 1)
                obj.write(line)
            for f in range(F):
                line = 'l {} {}\n'.format(2 * F + f + 1, f + 3 * F + 1)
                obj.write(line)
        else:
            F_rand = np.random.randint(0, F, F // 4)
            for f in F_rand:
                line = 'l {} {}\n'.format(f + 1, f + F + 1)
                obj.write(line)
        obj.close()

    def save_mesh(self, file_name, field_scale=1, V1=None, V2=None, save_all=False):
        geo.save_mesh_obj(self.optimized_vertices(), self._faces, file_name, overwrite=True)
        if V1 is None:
            self.save_directions_obj(file_name, scale=field_scale, save_all=save_all)
        else:
            self.save_directions_obj(file_name, scale=field_scale,
                                     V1=V1, V2=V2, save_all=save_all)

    # -------------------------------------------------------------------------
    # CONSTRAINTS
    # -------------------------------------------------------------------------

    def orthogonality_constraints(self):
        w = self.get_weight('orthogonal')
        N = self.N
        X = self.X
        V = len(self._vertices)
        F = len(self._faces)
        f = np.arange(F)
        x0 = self._faces[:, 0]
        y0 = self._faces[:, 0] + V
        z0 = self._faces[:, 0] + 2 * V
        x1 = self._faces[:, 1]
        y1 = self._faces[:, 1] + V
        z1 = self._faces[:, 1] + 2 * V
        x2 = self._faces[:, 2]
        y2 = self._faces[:, 2] + V
        z2 = self._faces[:, 2] + 2 * V
        a = self._x1[:, 0] * self._x2[:, 0]
        b = self._x1[:, 0] * self._x2[:, 1] + self._x1[:, 1] * self._x2[:, 0]
        c = self._x1[:, 1] * self._x2[:, 1]
        i = np.hstack((f, f, f, f, f, f, f, f, f, f, f, f,
                       f, f, f, f, f, f, f, f, f, f, f, f,
                       f, f, f, f, f, f, f,
                       f, f, f, f, f, f, f,
                       f, f, f, f, f, f, f,
                       ))
        j = np.hstack((x0, x1, x0, x1, y0, y1, y0, y1, z0, z1, z0, z1,
                       x0, x2, x0, x2, y0, y2, y0, y2, z0, z2, z0, z2,
                       x1, x2, x2, x0, x1, x0, x0,
                       y1, y2, y2, y0, y1, y0, y0,
                       z1, z2, z2, z0, z1, z0, z0,
                       ))
        data = np.hstack((2 * a * X[x0], 2 * a * X[x1], -2 * a * X[x1], -2 * a * X[x0],
                          2 * a * X[y0], 2 * a * X[y1], -2 * a * X[y1], -2 * a * X[y0],
                          2 * a * X[z0], 2 * a * X[z1], -2 * a * X[z1], -2 * a * X[z0],
                          2 * c * X[x0], 2 * c * X[x2], -2 * c * X[x2], -2 * c * X[x0],
                          2 * c * X[y0], 2 * c * X[y2], -2 * c * X[y2], -2 * c * X[y0],
                          2 * c * X[z0], 2 * c * X[z2], -2 * c * X[z2], -2 * c * X[z0],
                          b * X[x2], b * X[x1], -b * X[x0], -b * X[x2], -b * X[x0], -b * X[x1], 2 * b * X[x0],
                          b * X[y2], b * X[y1], -b * X[y0], -b * X[y2], -b * X[y0], -b * X[y1], 2 * b * X[y0],
                          b * X[z2], b * X[z1], -b * X[z0], -b * X[z2], -b * X[z0], -b * X[z1], 2 * b * X[z0],
                          )) * w
        r = np.hstack((a * (X[x0] ** 2 + X[x1] ** 2 - 2 * X[x1] * X[x0] +
                            X[y0] ** 2 + X[y1] ** 2 - 2 * X[y1] * X[y0] +
                            X[z0] ** 2 + X[z1] ** 2 - 2 * X[z1] * X[z0]) +
                       c * (X[x0] ** 2 + X[x2] ** 2 - 2 * X[x2] * X[x0] +
                            X[y0] ** 2 + X[y2] ** 2 - 2 * X[y2] * X[y0] +
                            X[z0] ** 2 + X[z2] ** 2 - 2 * X[z2] * X[z0]) +
                       b * (X[x2] * X[x1] - X[x0] * X[x2] - X[x0] * X[x1] + X[x0] ** 2 +
                            X[y2] * X[y1] - X[y0] * X[y2] - X[y0] * X[y1] + X[y0] ** 2 +
                            X[z2] * X[z1] - X[z0] * X[z2] - X[z0] * X[z1] + X[z0] ** 2)
                       )) * w
        H = sparse.coo_matrix((data, (i, j)), shape=(F, N))
        self.add_iterative_constraint(H, r, 'Orthogonal')

    def closeness_constraints(self):
        w = self.get_weight('closeness')
        N = self.N
        V = 3 * len(self._vertices)
        v = np.arange(V)
        data = np.ones(V) * w
        r = self._vertices.flatten('F') * w
        H = sparse.coo_matrix((data, (v, v)), shape=(V, N))
        self.add_iterative_constraint(H, r, 'Closeness')

    def mesh_fairness(self):
        w = self.get_weight('mesh_fairness')
        N = self.N
        V = len(self._vertices)
        bound = geo.boundary_vertices(self._halfedges)
        v0, vj, l = geo.vertices_ring_vertices(self._halfedges, order=True, return_lengths=True)
        inner = np.invert(np.in1d(v0, bound))
        v0 = v0[inner]
        vj = vj[inner]
        i1 = v0
        j1 = vj
        data1 = - np.ones(v0.shape[0]) / l[v0]
        i2 = np.unique(v0)
        j2 = np.unique(v0)
        data2 = np.ones(j2.shape[0])
        i = np.hstack((i1, i2))
        j = np.hstack((j1, j2))
        data = np.hstack((data1, data2)) * w
        W = int(np.amax(i)) + 1
        Kx = sparse.coo_matrix((data, (i, j)), shape=(W, N))
        j = V + j
        Ky = sparse.coo_matrix((data, (i, j)), shape=(W, N))
        j = V + j
        Kz = sparse.coo_matrix((data, (i, j)), shape=(W, N))
        K = sparse.vstack([Kx, Ky, Kz])
        s = np.zeros(int(3 * W))
        self.add_constant_fairness(K, s)

    def boundary_fairness(self, ):
        w = self.get_weight('mesh_fairness')
        N = self.N
        V = len(self._vertices)
        corners = geo.mesh_corners(self._vertices, self._halfedges, 0.9)
        print(corners)
        boundaries = geo.boundary_contiguous_vertices(self._halfedges, corners)
        K = sparse.coo_matrix(([0], ([0], [0])), shape=(1, N))
        for boundary in boundaries:
            mask = np.invert(np.in1d(boundary, corners))
            v0 = boundary[mask]
            W = v0.shape[0]
            vn = np.roll(boundary, 1)[mask]
            vp = np.roll(boundary, -1)[mask]
            one = np.ones(W)
            v = np.arange(W)
            i = np.hstack((v, v, v, W + v, W + v, W + v, 2 * W + v, 2 * W + v, 2 * W + v))
            j = np.hstack((v0, vp, vn, V + v0, V + vp, V + vn, 2 * V + v0, 2 * V + vp, 2 * V + vn))
            data = np.hstack((2 * one, -one, -one))
            data = np.hstack((data, data, data)) * w
            Ki = sparse.coo_matrix((data, (i, j)), shape=(3 * W, N))
            K = sparse.vstack([K, Ki])
        s = np.zeros((K.shape[0]))
        self.add_constant_fairness(K, s)

    # -------------------------------------------------------------------------
    # BUILD
    # -------------------------------------------------------------------------

    def build_iterative_constraints(self):
        pass
        self.orthogonality_constraints()
        self.closeness_constraints()

    def build_constant_fairness(self):
        self.mesh_fairness()
        self.boundary_fairness()


if __name__ == '__main__':
    V, F = geo.read_mesh('script/tri_dome_flat.obj')
    H = geo.halfedges(F)
    V, H = geo.loop_subdivision(V, H)
    F = np.array(geo.faces_list(H))
    geo.save_mesh_obj(V, H, 'rem1_0')

    v1 = np.zeros((len(F), 3))
    v1[:, 1] = 1
    v2 = np.zeros((len(F), 3))
    v2[:, 0] = 1

    opt = OrthoOptimizer(V, F, v1, v2)
    opt.iterations = 0
    opt.step = 0.3
    opt.optimize()
    opt.save_mesh('rem1')

    v1, v2 = opt.vectors()

    V1 = opt.optimized_vertices()
    B = geo.faces_centroid(V1, H)
    plotter = geo.plotter()
    plotter.plot_edges(V, F)
    plotter.plot_edges(V1, F, color='r')
    plotter.plot_vectors(v1, anchor=B, position='center')
    plotter.plot_vectors(v2, anchor=B, position='center')
    plotter.show()
