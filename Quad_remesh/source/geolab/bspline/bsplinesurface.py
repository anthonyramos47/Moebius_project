# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

from scipy import spatial

# -----------------------------------------------------------------------------

from geolab.utilities import bsplineutilities

from geolab.utilities.stringutilities import make_filepath

from geolab.utilities.arrayutilities import normalize

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#                            B-SPLINE SURFACE CLASS
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


class BsplineSurface(object):
    """B-spline surface class.

    Attributes
    ----------
    name : str
        Name of the surface.
    control_points : (C, 3) array
        Array of control points.
    u_degree : int
        Degree of the surface in the u direction.
    v_degree : int
        Degree of the surface in the v direction.
    u_knot_vector : (Cu + u_degree + 1) array
        Knot vector in the u direction.
    v_knot_vector : (Cv + v_degree + 1) array
        Knot vector in the v direction.
    Cu : int
        Number of control points in the u direction.
    Cv : int
        Number of control points in the v direction.
    C : int
        Total number of control points.
    u_sampling : int
        Number of points to sample in the u direction.
    v_sampling : int
        Number of points to sample in the v direction.
    """

    def __init__(self, u_degree=3, v_degree=3, control_points=None,
                 u_knot_vector=None, v_knot_vector=None, file_name=None):

        self.name = 'bspline-surface'

        self._u_degree = u_degree

        self._v_degree = v_degree

        self._control_points = control_points

        self._control_points_index = None

        self._u_knot_vector = u_knot_vector

        self._v_knot_vector = v_knot_vector

        self._u_sampling = 100

        self._v_sampling = 100

        self._u_range = None

        self._v_range = None

        self._u = None

        self._v = None

        self.plot_mode = 'surface'

        self._sealed = [False, False]

        self._kdtree = None

        if file_name is not None:
            self.read_obj_file(file_name)

        print(self)

    # -------------------------------------------------------------------------

    def __str__(self):
        n = 'B-spline surface: '
        info = 'degree = ({},{}), number of control points = '
        out = n + info.format(self._u_degree, self._v_degree)
        degree = '({}'
        U = self.Cu
        V = self.Cv
        if self._sealed[0]:
            U -= self.u_degree
            degree += 'c'
        degree += ',{}'
        if self._sealed[1]:
            V -= self.v_degree
            degree += 'c'
        degree += ')'
        return out + degree.format(U, V)

    def copy_bspline(self):
        B = BsplineSurface(None, self.u_degree, self.v_degree, np.copy(self.control_points),
                           np.copy(self._u_knot_vector), np.copy(self._v_knot_vector))
        return B

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    @property
    def type(self):
        return 'Bspline_Surface'

    @property
    def u_sealed(self):
        return self._sealed[0]

    @property
    def v_sealed(self):
        return self._sealed[1]

    @property
    def number_of_control_points(self):
        U = self.Cu
        V = self.Cv
        if self._sealed[0]:
            U -= self.u_degree
        if self._sealed[1]:
            V -= self.v_degree
        return (U, V)

    @property
    def C(self):
        return len(self._control_points)

    @property
    def Cu(self):
        return len(self._u_knot_vector) - self._u_degree - 1

    @property
    def Cv(self):
        return len(self._v_knot_vector) - self._v_degree - 1

    @property
    def S(self):
        return self.u_sampling * self.v_sampling

    @property
    def u_degree(self):
        return self._u_degree

    @property
    def v_degree(self):
        return self._v_degree

    @property
    def u(self):
        if self._u is None:
            self._u = np.linspace(self._u_range[0], self._u_range[1],
                                  self.u_sampling)
        return self._u

    @u.setter
    def u(self, u):
        if type(u) == int or type(u) == float:
            u = np.array(u)
        self._u = u
        self._u_sampling = len(u)

    @property
    def v(self):
        if self._v is None:
            self._v = np.linspace(self._v_range[0], self._v_range[1],
                                  self.v_sampling)
        return self._v

    @v.setter
    def v(self, v):
        if type(v) == int or type(v) == float:
            v = np.array(v)
        self._v = v
        self._v_sampling = len(v)

    @property
    def u_sampling(self):
        return self._u_sampling

    @u_sampling.setter
    def u_sampling(self, nu):
        self._u_sampling = int(nu)
        self._u = None

    @property
    def v_sampling(self):
        return self._v_sampling

    @v_sampling.setter
    def v_sampling(self, nv):
        self._v_sampling = int(nv)
        self._v = None

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, control_points):
        self._control_points = control_points

    @property
    def d(self):
        return self.control_points.shape[1]

    @property
    def V(self):
        if self.plot_mode == 'surface':
            return self.S
        else:
            return self.C

    @property
    def E(self):
        if self.plot_mode == 'surface':
            E = ((self.u_sampling - 1) * self.v_sampling +
                 (self.v_sampling - 1) * self.u_sampling)
            return E
        else:
            return (self.Cu - 1) * self.Cv + (self.Cv - 1) * self.Cu

    @property
    def vertices(self):
        if self.plot_mode == 'surface':
            return self.points()
        else:
            return self.control_points

    @property
    def degree(self):
        return self._u_degree, self._v_degree

    @property
    def sampling(self):
        return (self.u_sampling, self.v_sampling)

    @sampling.setter
    def sampling(self, sampling):
        self.u_sampling = sampling[0]
        self.v_sampling = sampling[1]

    # -------------------------------------------------------------------------
    #                             Reading - writing
    # -------------------------------------------------------------------------

    def read_obj_file(self, file_name):
        file_name = str(file_name)
        self.name = file_name.split('.')[0]
        obj_file = open(file_name, encoding='utf-8')
        vertices_list = []
        read = 0
        control = False
        for l in obj_file:
            splited_line = l.split(' ')
            if splited_line[0] == 'v':
                split_x = splited_line[1].split('\n')
                x = float(split_x[0])
                split_y = splited_line[2].split('\n')
                y = float(split_y[0])
                split_z = splited_line[3].split('\n')
                try:
                    z = float(split_z[0])
                except ValueError:
                    print('WARNING: disable line wrap when saving .obj')
                vertices_list.append([x, y, z])
            elif splited_line[0] == 'deg':
                if len(splited_line) == 3:
                    self._u_degree = int(splited_line[1])
                    self._v_degree = int(splited_line[2])
                    read += 1
            if read == 1:
                if splited_line[0] == 'surf':
                    self._u_range = np.array((float(splited_line[1]),
                                              float(splited_line[2])))
                    self._v_range = np.array((float(splited_line[3]),
                                              float(splited_line[4])))
                    indices = []
                    for i in range(len(splited_line) - 5):
                        indices.append(int(splited_line[i + 5]) - 1)
                if splited_line[0] == 'parm':
                    if splited_line[1] == 'u':
                        u_knot = []
                        for i in range(len(splited_line) - 2):
                            u_knot.append(float(splited_line[i + 2]))
                    elif splited_line[1] == 'v':
                        v_knot = []
                        for i in range(len(splited_line) - 2):
                            v_knot.append(float(splited_line[i + 2]))
                        read += 1
                        control = True
        if not control:
            raise ValueError('unable to read the file')
        self._control_points = np.array(vertices_list)
        self._control_points_index = np.array(indices, 'i')
        self._u_knot_vector = np.array(u_knot)
        self._v_knot_vector = np.array(v_knot)
        self._seal()

    def make_obj_file(self, file_name, overwrite=False):
        """Save the bspline surface as OBJ file.

        Parameters
        ----------
        file_name : str
            The path of the OBJ file to be created, without extension.
        overwrite : bool
            If `False` (default), when the path already exists, a sequential
            number is added to file_name. If `True`, existing paths are
            overwritten.

        Returns
        -------
        str
            The path of the newly generated OBJ file (without extension).
        """
        path = make_filepath(file_name, 'obj', overwrite)
        obj = open(path, 'w')
        line = 'o {}\n'.format(file_name)
        obj.write(line)
        for c in range(self.C):
            ci = self.control_points[c]
            line = 'v {} {} {}\n'.format(ci[0], ci[1], ci[2])
            obj.write(line)
        line = 'cstype bspline\n'
        obj.write(line)
        line = 'deg {} {}\n'.format(self._u_degree, self._v_degree)
        obj.write(line)
        line = 'surf {} {} {} {}'.format(self._u_range[0], self._u_range[1],
                                         self._v_range[0], self._v_range[1])
        for c in self._control_points_index:
            line += ' {}'.format(c + 1)
        line += '\n'
        obj.write(line)
        line = 'parm u'
        for u in self._u_knot_vector:
            line += ' {}'.format(u)
        line += '\n'
        obj.write(line)
        line = 'parm v'
        for v in self._v_knot_vector:
            line += ' {}'.format(v)
        line += '\n'
        line += 'end'
        obj.write(line)
        obj.close()
        out = 'Obj saved in {}'.format(path)
        print(out)
        return path.split('.')[0]

    # -------------------------------------------------------------------------
    #                                  Seal
    # -------------------------------------------------------------------------

    def _seal_check(self):
        c = self._control_points_index
        c = np.reshape(self._control_points_index, (self.Cu, self.Cv), 'F')
        if len(np.unique(c[:, 0])) < len(c[:, 0]):
            self._sealed[0] = True
        if len(np.unique(c[0, :])) < len(c[0, :]):
            self._sealed[1] = True

    def _seal(self, epsilon=1e-5):
        v = self._control_points_index
        if len(np.unique(v)) < len(v):
            self._seal_check()
        else:
            u_seal = 0
            v_seal = 0
            ud = self.u_degree
            vd = self.v_degree
            for i in range(self.d):
                G = np.reshape(self.control_points[v, i], (self.Cu, self.Cv), 'F')
                u_seal += np.sum(np.abs(G[:ud, :] - G[-ud:, :]))
                v_seal += np.sum(np.abs(G[:, :vd] - G[:, -vd:]))
            ind = np.copy(np.reshape(v, (self.Cu, self.Cv), 'F'))
            c_map = np.reshape(v, (self.Cu, self.Cv), 'F')
            if u_seal < epsilon:
                ind[-ud:, :] = ind[:ud, :]
                c_map = c_map[:-ud, :]
                ind[:, np.arange(self.Cv)] -= ud * np.arange(self.Cv)
                self._sealed[0] = True
            if v_seal < epsilon:
                ind[:, -vd:] = ind[:, :vd]
                c_map = c_map[:, :-vd]
                # may have problems with non arange indices v
                ind -= np.min(ind[:, :vd])
                self._sealed[1] = True
            if self._sealed[0] or self._sealed[1]:
                self._control_points_index = ind.flatten('F')
                self._control_points = self._control_points[c_map.flatten('F')]

    # -------------------------------------------------------------------------
    #                                Evaluation
    # -------------------------------------------------------------------------

    def basis_functions(self):
        Nu = bsplineutilities.bspline_basis_functions(self.u, self.u_degree,
                                                      self._u_knot_vector)
        Nv = bsplineutilities.bspline_basis_functions(self.v, self.v_degree,
                                                      self._v_knot_vector)
        return Nu, Nv

    def derivative_basis_functions(self, d=1):
        Du = bsplineutilities.bspline_derivative_basis_functions(self.u,
                                                                 self.u_degree,
                                                                 self._u_knot_vector,
                                                                 d=d)
        Dv = bsplineutilities.bspline_derivative_basis_functions(self.v,
                                                                 self.v_degree,
                                                                 self._v_knot_vector,
                                                                 d=d)
        return Du, Dv

    def points(self):
        Nu, Nv = self.basis_functions()
        points = []
        for i in range(self.d):
            v = self._control_points_index
            G = np.reshape(self.control_points[v, i], (self.Cu, self.Cv), 'F')
            Pi = np.einsum('ij,ui,vj->uv', G, Nu, Nv)
            points.append(np.reshape(Pi, self.S, 'F'))
        return np.column_stack(points)

    def u_lines(self):
        u = np.unique(self._u_knot_vector)
        Nu = bsplineutilities.bspline_basis_functions(u, self.u_degree,
                                                      self._u_knot_vector)
        Nv = bsplineutilities.bspline_basis_functions(self.v, self.v_degree,
                                                      self._v_knot_vector)
        uc = []
        for i in range(self.d):
            v = self._control_points_index
            G = np.reshape(self.control_points[v, i], (self.Cu, self.Cv), 'F')
            P = np.einsum('ij,ui,vj->uv', G, Nu, Nv)
            uc.append(P)
        uc = np.dstack(uc)
        return uc

    def v_lines(self):
        v = np.unique(self._v_knot_vector)
        Nu = bsplineutilities.bspline_basis_functions(self.u, self.u_degree,
                                                      self._u_knot_vector)
        Nv = bsplineutilities.bspline_basis_functions(v, self.v_degree,
                                                      self._v_knot_vector)
        vc = []
        for i in range(self.d):
            v = self._control_points_index
            G = np.reshape(self.control_points[v, i], (self.Cu, self.Cv), 'F')
            P = np.einsum('ij,ui,vj->uv', G, Nu, Nv)
            vc.append(P.T)
        vc = np.dstack(vc)
        return vc

    def uv_points(self):
        u = np.tile(self.u, self.v_sampling)
        v = np.repeat(self.v, self.u_sampling)
        return np.column_stack((u, v))

    def first_derivatives(self):
        Nu, Nv = self.basis_functions()
        Nuu, Nvv = self.derivative_basis_functions(d=1)
        Du = np.empty((self.S, 0))
        Dv = np.empty((self.S, 0))
        for i in range(self.d):
            v = self._control_points_index
            G = np.reshape(self.control_points[v, i], (self.Cu, self.Cv), 'F')
            Ui = np.einsum('ij,ui,vj->uv', G, Nuu, Nv)
            Vi = np.einsum('ij,ui,vj->uv', G, Nu, Nvv)
            Du = np.column_stack((Du, np.reshape(Ui, self.S, 'F')))
            Dv = np.column_stack((Dv, np.reshape(Vi, self.S, 'F')))
        return Du, Dv

    def second_derivatives(self, return_first=True):
        Nu, Nv = self.basis_functions()
        Nu_u, Nv_v = self.derivative_basis_functions(d=1)
        Nu_uu, Nv_vv = self.derivative_basis_functions(d=2)
        if return_first:
            Du = np.empty((self.S, 0))
            Dv = np.empty((self.S, 0))
        Duu = np.empty((self.S, 0))
        Duv = np.empty((self.S, 0))
        Dvv = np.empty((self.S, 0))
        for i in range(self.d):
            v = self._control_points_index
            G = np.reshape(self.control_points[v, i], (self.Cu, self.Cv), 'F')
            if return_first:
                Du_i = np.einsum('ij,ui,vj->uv', G, Nu_u, Nv)
                Dv_i = np.einsum('ij,ui,vj->uv', G, Nu, Nv_v)
                Du = np.column_stack((Du, np.reshape(Du_i, self.S, 'F')))
                Dv = np.column_stack((Dv, np.reshape(Dv_i, self.S, 'F')))
            Duu_i = np.einsum('ij,ui,vj->uv', G, Nu_uu, Nv)
            Duv_i = np.einsum('ij,ui,vj->uv', G, Nu_u, Nv_v)
            Dvv_i = np.einsum('ij,ui,vj->uv', G, Nu, Nv_vv)
            Duu = np.column_stack((Duu, np.reshape(Duu_i, self.S, 'F')))
            Duv = np.column_stack((Duv, np.reshape(Duv_i, self.S, 'F')))
            Dvv = np.column_stack((Dvv, np.reshape(Dvv_i, self.S, 'F')))
        if return_first:
            return Du, Dv, Duu, Duv, Dvv
        else:
            return Duu, Duv, Dvv

    def first_fundamental_form(self, Du=None, Dv=None):
        if Du is None or Dv is None:
            Du, Dv = self.first_derivatives()
        E = np.einsum('ij,ij->i', Du, Du)
        F = np.einsum('ij,ij->i', Du, Dv)
        G = np.einsum('ij,ij->i', Dv, Dv)
        return E, F, G

    def fundamental_forms(self, return_curvatures=True):
        Du, Dv, Duu, Duv, Dvv = self.second_derivatives()
        E = np.einsum('ij,ij->i', Du, Du)
        F = np.einsum('ij,ij->i', Du, Dv)
        G = np.einsum('ij,ij->i', Dv, Dv)
        n = normalize(np.cross(Du, Dv))
        L = np.einsum('ij,ij->i', n, Duu)
        M = np.einsum('ij,ij->i', n, Duv)
        N = np.einsum('ij,ij->i', n, Dvv)
        if return_curvatures:
            K = (L * N - M ** 2) / (E * G - F ** 2)
            H = (L * G - 2 * M * F + N * E) / (2 * (E * G - F ** 2))
            return E, F, G, L, M, N, K, H
        else:
            return E, F, G, L, M, N

    def curvatures(self):
        E, F, G, L, M, N, K, H = self.fundamental_forms()
        k1 = H + np.abs(H ** 2 - K) ** .5
        k2 = H - np.abs(H ** 2 - K) ** .5
        return K, H, k1, k2

    def closest_point(self, points):
        kdtree = spatial.cKDTree(self.points())
        return kdtree.query(points)[1]

    def normals(self):
        Du, Dv = self.first_derivatives()
        N = np.cross(Du, Dv)
        N = normalize(N)
        return N

    # -------------------------------------------------------------------------
    #                                 Iterators
    # -------------------------------------------------------------------------

    def point_basis_functions_iterators(self):
        c = self._control_points_index
        i = np.tile(np.arange(self.Cu), self.Cv)
        j = np.repeat(np.arange(self.Cv), self.Cu)
        return c, i, j

    def uv_sampling_iterators(self):
        """
        V[s] = P[c] Nu[u,i] Nv[v,j]
        """
        c, i, j = self.point_basis_functions_iterators()
        s = np.repeat(np.arange(self.S), len(c))
        u = np.tile(np.arange(self.u_sampling), self.v_sampling)
        v = np.repeat(np.arange(self.v_sampling), self.u_sampling)
        u = np.repeat(u, self.Cu * self.Cv)
        v = np.repeat(v, self.Cu * self.Cv)
        c = np.tile(c, self.S)
        i = np.tile(i, self.S)
        j = np.tile(j, self.S)
        return s, u, v, c, i, j

    def uv_central_differences_iterators(self, return_uv_delta=False,
                                         include_boundary=True):
        U = self.u_sampling
        V = self.v_sampling
        M = np.arange(U * V)
        M = M.reshape((U, V), order='F')
        if include_boundary:
            M = np.column_stack((M[:, 0], M, M[:, -1]))
            M = np.vstack((M[0, :], M, M[-1, :]))
        up = M[2:, 1:-1].flatten('F')
        um = M[:-2, 1:-1].flatten('F')
        vp = M[1:-1, 2:].flatten('F')
        vm = M[1:-1, :-2].flatten('F')
        p = M[1:-1, 1:-1].flatten('F')
        if return_uv_delta:
            du = np.hstack((self.u[1] - self.u[0], self.u[2:] - self.u[:-2],
                            self.u[-1] - self.u[-2]))
            dv = np.hstack((self.v[1] - self.v[0], self.v[2:] - self.v[:-2],
                            self.v[-1] - self.v[-2]))
            du = np.tile(du, V)
            dv = np.tile(dv, U)
            return p, up, um, vp, vm, du, dv
        else:
            return p, up, um, vp, vm

    # def uv_second_differences_iterators(self):
    #     U = self.u_sampling
    #     V = self.v_sampling
    #     M = np.arange(U*V)
    #     M = M.reshape((U,V), order='F')
    #     s = M[2:-2,2:-2].flatten('F')
    #     up = M[3:-1,2:-2].flatten('F')
    #     um = M[1:-3,2:-2].flatten('F')
    #     vp = M[2:-2,3:-1].flatten('F')
    #     vm = M[2:-2,1:-3].flatten('F')
    #     upp = M[4:,2:-2].flatten('F')
    #     umm = M[:-4,2:-2].flatten('F')
    #     vpp = M[2:-2,4:].flatten('F')
    #     vmm = M[2:-2,:-4].flatten('F')
    #     return s, upp, up, um, umm, vpp, vp, vm, vmm

    def uv_central_differences(self, f, return_uv_delta=False,
                               include_boundary=True):
        p, up, um, vp, vm, du, dv = self.uv_central_differences_iterators(True)
        Du = (f[up] - f[um]) / du
        Dv = (f[vp] - f[vm]) / dv
        if return_uv_delta:
            return Du, Dv, du, dv
        else:
            return Du, Dv

    # -------------------------------------------------------------------------
    #                                   Plot
    # -------------------------------------------------------------------------

    def cell_arrays(self):
        if self.plot_mode == 'surface':
            V = self.v_sampling
            U = self.u_sampling
        else:
            V = self.Cv
            U = self.Cu
        M = np.arange(V * U, dtype=np.int)
        M = np.reshape(M, (V, U))
        Q = np.zeros((V - 1, U - 1, 4), dtype=np.int)
        Q[:, :, 3] = M[:M.shape[0] - 1, :M.shape[1] - 1]
        Q[:, :, 2] = M[:M.shape[0] - 1, 1:]
        Q[:, :, 1] = M[1:, 1:]
        Q[:, :, 0] = M[1:, :M.shape[1] - 1]
        Q = np.reshape(Q, ((V - 1) * (U - 1), 4), 'C')
        Q = np.column_stack((4 * np.ones((V - 1) * (U - 1)), Q))
        Q = np.reshape(Q, Q.shape[0] * Q.shape[1], 'C')
        self.F = (V - 1) * (U - 1)
        types = np.repeat(1, self.F)
        return Q, types

    def edge_vertices(self, mode='control'):
        if mode == 'surface':
            V = self.v_sampling
            U = self.u_sampling
            M = np.arange(V * U, dtype=np.int)
        else:
            V = self.Cv
            U = self.Cu
            M = self._control_points_index
        M = np.reshape(M, (V, U))
        v1 = np.reshape(M[:, :-1], (V * (U - 1)), 'F')
        v2 = np.reshape(M[:, 1:], (V * (U - 1)), 'F')
        v1 = np.hstack((v1, np.reshape(M.T[:, :-1], ((V - 1) * U), 'F')))
        v2 = np.hstack((v2, np.reshape(M.T[:, 1:], ((V - 1) * U), 'F')))
        # self.E = len(v1)
        return v1, v2

    def face_diagonals(self):
        V = self.Cv
        U = self.Cu
        M = self._control_points_index
        M = np.reshape(M, (V, U))
        v1 = M[:-1, :-1].flatten('F')
        v2 = M[1:, 1:].flatten('F')
        v3 = M[:-1, 1:].flatten('F')
        v4 = M[1:, :-1].flatten('F')
        d1 = np.hstack((v1, v3))
        d2 = np.hstack((v2, v4))
        return d1, d2

    def inner_vertices(self):
        V = self.v_sampling
        U = self.u_sampling
        M = np.arange(V * U, dtype=np.int)
        M = np.reshape(M, (V, U))
        inner = M[1:-1, 1:-1]
        return inner.flatten()

    # -------------------------------------------------------------------------
    #                            Boundaries
    # -------------------------------------------------------------------------

    def boundary_control_points(self):
        c = np.reshape(self._control_points_index, (self.Cu, self.Cv), 'F')
        boundary = np.array([], 'i')
        if not self._sealed[0]:
            boundary = np.hstack((boundary, c[0, :], c[-1, :]))
        if not self._sealed[1]:
            boundary = np.hstack((boundary, c[:, 0], c[:, -1]))
        return np.unique(boundary)

    def boundary_control_curves(self):
        c = np.reshape(self._control_points_index, (self.Cu, self.Cv), 'F')
        curves = []
        if not self._sealed[0]:
            curves.append(c[0, :])
            curves.append(c[-1, :])
        if not self._sealed[1]:
            curves.append(c[:, 0])
            curves.append(c[:, -1])
        return curves

    def control_corners(self):
        c = np.reshape(self._control_points_index, (self.Cu, self.Cv), 'F')
        return [c[0, 0], c[0, -1], c[-1, -1], c[-1, 0]]

    def control_points_curves_iterators(self):
        c = np.reshape(self._control_points_index, (self.Cu, self.Cv), 'F')
        cu = (c[1:-1, :]).flatten('F')
        cu, iu = np.unique(cu, return_index=True)
        up = ((c[2:, :]).flatten('F'))[iu]
        um = ((c[:-2, :]).flatten('F'))[iu]
        cv = (c[:, 1:-1]).flatten('F')
        cv, iv = np.unique(cv, return_index=True)
        vp = ((c[:, 2:]).flatten('F'))[iv]
        vm = ((c[:, :-2]).flatten('F'))[iv]
        c = np.hstack((cu, cv))
        cp = np.hstack((up, vp))
        cm = np.hstack((um, vm))
        return c, cp, cm

    # -------------------------------------------------------------------------
    #                             Utilities
    # -------------------------------------------------------------------------

    def scale(self, factor):
        self._control_points *= factor
        self._u_knot_vector *= factor
        self._v_knot_vector *= factor
        self._u_range *= factor
        self._v_range *= factor
        self._u = None
        self._v = None

    def uv_faces(self):
        V = self.v_sampling
        U = self.u_sampling
        M = np.arange(V * U, dtype=np.int)
        M = np.reshape(M, (V, U))
        Q = np.zeros((V - 1, U - 1, 4), dtype=np.int)
        Q[:, :, 3] = M[:M.shape[0] - 1, :M.shape[1] - 1]
        Q[:, :, 2] = M[:M.shape[0] - 1, 1:]
        Q[:, :, 1] = M[1:, 1:]
        Q[:, :, 0] = M[1:, :M.shape[1] - 1]
        Q = np.reshape(Q, ((V - 1) * (U - 1), 4), 'C')
        return Q

    def uv_triangles(self):
        Q = self.uv_faces()
        T = np.vstack((Q[:, [0, 1, 2]], Q[:, [2, 3, 0]]))
        return T

    def flip_orientation(self):
        C = []
        for i in range(self.d):
            v = self._control_points_index
            G = np.reshape(self.control_points[v, i], (self.Cu, self.Cv), 'F')
            G = G.T.flatten('F')
            C.append(G)
        self._control_points = np.column_stack(C)

        v_degree = self._u_degree
        u_degree = self._v_degree
        self._u_degree = u_degree
        self._v_degree = v_degree

        u_knot_vector = self._v_knot_vector
        v_knot_vector = self._u_knot_vector
        self._u_knot_vector = u_knot_vector
        self._v_knot_vector = v_knot_vector

        u_sampling = self._v_sampling
        v_sampling = self._u_sampling
        self._u_sampling = u_sampling
        self._v_sampling = v_sampling

        u_range = self._v_range
        v_range = self._u_range
        self._u_range = u_range
        self._v_range = v_range

        u = self._v
        v = self._u
        self._u = u
        self._v = v

        self._sealed = [self._sealed[1], self._sealed[0]]

    # -------------------------------------------------------------------------
    #                              Rebuild
    # -------------------------------------------------------------------------

    def rebuild(self, u_degree=None, v_degree=None, u_points=None,
                v_points=None):
        if u_degree is None:
            u_degree = self.u_degree
        if v_degree is None:
            v_degree = self.v_degree
        if u_points is None:
            u_points = self.Cu
        elif self._sealed[0]:
            u_points = max(u_points, u_degree)
            u_points += u_degree
        else:
            u_points = max(u_points, u_degree + 1)
        if v_points is None:
            v_points = self.Cv
        elif self._sealed[1]:
            v_points = max(v_points, v_degree)
            v_points += v_degree
        else:
            v_points = max(v_points, v_degree + 1)
        ind = np.reshape(np.arange(u_points * v_points), (u_points, v_points), 'F')
        u_knot = np.linspace(0, 1, u_points - u_degree + 1)
        if not self._sealed[0]:
            u_knot = np.hstack((np.repeat(0, u_degree), u_knot,
                                np.repeat(1, u_degree)))
        else:
            delta = u_knot[1] - u_knot[0]
            i = np.arange(u_degree) + 1
            u_knot = np.hstack((-np.flip(i) * delta, u_knot, 1 + i * delta))
            ind[-u_degree:, :] = ind[:u_degree, :]
            ind[:, np.arange(v_points)] -= u_degree * np.arange(v_points)
        v_knot = np.linspace(0, 1, v_points - v_degree + 1)
        if not self._sealed[1]:
            v_knot = np.hstack((np.repeat(0, v_degree), v_knot,
                                np.repeat(1, v_degree)))
        else:
            delta = v_knot[1] - v_knot[0]
            i = np.arange(v_degree) + 1
            v_knot = np.hstack((-np.flip(i) * delta, v_knot, 1 + i * delta))
            ind[:, -v_degree:] = ind[:, :v_degree]
            ind -= np.min(ind[:, :v_degree])
        P0 = self.points()
        ut = np.linspace(0, 1, self.u_sampling)
        vt = np.linspace(0, 1, self.v_sampling)
        Nu = bsplineutilities.bspline_basis_functions(ut, u_degree, u_knot)
        Nv = bsplineutilities.bspline_basis_functions(vt, v_degree, v_knot)
        self._control_points_index = ind.flatten('F')
        self._u_knot_vector = u_knot
        self._v_knot_vector = v_knot
        self._u_degree = u_degree
        self._v_degree = v_degree
        self._u = None
        self._v = None
        self._u_range = np.array([0, 1])
        self._v_range = np.array([0, 1])
        s, u, v, c, cu, cv = self.uv_sampling_iterators()
        Cn = np.max(ind) + 1
        X = []
        H = sparse.coo_matrix((Nu[u, cu] * Nv[v, cv], (s, c)), shape=(self.S, Cn))
        H = H.toarray()
        A = np.dot(H.T, H)
        for d in range(self.d):
            Xd = np.linalg.solve(A, np.dot(H.T, P0[:, d]))
            X.append(Xd)
        P = np.column_stack(X)
        self._control_points = P
        self.relax_knot_vector()
        print(self)

    def relax_knot_vector(self):
        Du, Dv = self.first_derivatives()
        fu = np.mean(np.linalg.norm(Du, axis=1))
        fv = np.mean(np.linalg.norm(Dv, axis=1))
        self._u_knot_vector = self._u_knot_vector * fu
        self._u_range = self._u_range * fu
        self._v_knot_vector = self._v_knot_vector * fv
        self._v_range = self._v_range * fv
        self._u = None
        self._v = None


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#                                   API
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


def read_bspline(file_name):
    """Read a B-spline surface OBJ file.

    Parameters
    ----------
    file_name : str
        Name of the file to read.

    Returns
    -------
    B-spline surface
        An instance of the BsplineSurface class.
    """
    return BsplineSurface(file_name=file_name)


def save_bspline(bspline, file_name, overwrite=False):
    """Save the bspline surface as OBJ file.

    Parameters
    ----------
    bspline : geolab.BsplineSurface
        The bspline surface to be saved.
    file_name : str
        The path of the OBJ file to be created, without extension.

    Optional Parameters
    -------------------
    overwrite : bool (default: False)
        If `False`, when the path already exists, a sequential
        number is added to file_name. If `True`, existing paths are
        overwritten.

    Returns
    -------
    str
        The path of the newly generated OBJ file (without extension).
    """
    return bspline.make_obj_file(file_name, overwrite=overwrite)


def bspline_surface(u_degree=3, v_degree=3, control_points=None,
                    u_knot_vector=None, v_knot_vector=None, file_name=None):
    """Generate a B-spline surface.

    Parameters
    ----------
    u_degree : int
        Degree of the surface in the u direction. Default is 3.
    v_degree : int
        Degree of the surface in the v direction. Default is 3.
    control_points : (C, 3) array
        Array of control points.
    u_knot_vector : array
        Knot vector in the u direction.
    v_knot_vector : array
        Knot vector in the v direction.
    file_name : str
        Name of the file to read.

    Returns
    -------
    B-spline surface
        An instance of the BsplineSurface class.
    """
    return BsplineSurface(u_degree, v_degree, control_points, u_knot_vector,
                          v_knot_vector, file_name)