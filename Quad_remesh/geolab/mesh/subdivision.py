
__author__ = 'Davide Pellis'

import numpy as np

from scipy import spatial

import geolab
from geolab.mesh.globalconnectivity import faces_size, vertices_ring_vertices, \
    faces_vertices, faces_list, edges_halfedges

from geolab.mesh.geometry import faces_centroid, edges_mid_point, mean_edge_length, \
    vertices_ring_barycenter, vertices_normal

from geolab.mesh.halfedges import halfedges

from geolab.mesh.boundary import boundary_edges, inner_halfedges, boundary_vertices, \
    mesh_corners, boundary_halfedges

# -----------------------------------------------------------------------------

from geolab.utilities.arrayutilities import sum_repeated, repeated_range

from geolab.mesh.globalconnectivity import faces_list

# -----------------------------------------------------------------------------


class Mesh(object):

    def __init__(self, vertices, halfedges):

        self.halfedges = np.copy(halfedges)

        self.vertices = np.copy(vertices)

        self._V = np.amax(self.halfedges[:, 0] + 1)

        self._F = np.amax(self.halfedges[:, 1] + 1)

        self._E = np.amax(self.halfedges[:, 5] + 1)

        self._h_trash = None

        self._v_trash = None

        self._f_trash = None

        self._e_trash = None

        self.corner_tolerance = 0.7

        self.fixed_boundary = False

    # -------------------------------------------------------------------------
    #                                 Update
    # -------------------------------------------------------------------------

    def _topology_update(self):
        self._V = len(self.vertices)#np.amax(self.halfedges[:, 0] + 1)
        self._F = np.amax(self.halfedges[:, 1] + 1)
        self._E = np.amax(self.halfedges[:, 5] + 1)

    def _update_vertices(self, vertices):
        self.vertices = vertices

    def _update_faces(self, faces):
        self.halfedges = halfedges(faces)
        self._topology_update()

    def _update_halfedges(self, halfedges):
        self.halfedges = halfedges
        self._topology_update()

    def is_triangular_mesh(self):
        if len(np.where(faces_size(self.halfedges) != 3)[0] > 0):
            return False
        else:
            return True

    # -------------------------------------------------------------------------
    #                              Subdivision
    # -------------------------------------------------------------------------

    def loop(self, fixed_vertices=[]):
        if not self.is_triangular_mesh():
            return
        V = self._V
        H = self.halfedges
        v_fix = self.vertices[fixed_vertices]
        _, h1 = np.unique(H[:, 1], True)
        h1 = np.delete(h1, np.where(H[h1, 1] == -1))
        h2 = H[h1, 2]
        h3 = H[h1, 3]
        F0 = np.array((H[h1, 5] + V, H[h2, 5] + V, H[h3, 5] + V)).T
        F1 = np.array((H[h1, 0], H[h1, 5] + V, H[H[h1, 3], 5] + V)).T
        F2 = np.array((H[h2, 0], H[h2, 5] + V, H[H[h2, 3], 5] + V)).T
        F3 = np.array((H[h3, 0], H[h3, 5] + V, H[H[h3, 3], 5] + V)).T
        new_faces = np.vstack((F0, F1, F2, F3))
        v, vj, li = vertices_ring_vertices(H, sort=True, return_lengths=True)
        c = 1./li * (5/8 - (3/8 + 1/4 * np.cos(2 * np.pi * li ** (-1.))) ** 2)
        d = 1 - li * c
        c = np.array([c, c, c]).T
        d = np.array([d, d, d]).T
        ring = sum_repeated(self.vertices[vj], v)
        V0 = c * ring + d * self.vertices
        V0[fixed_vertices] = v_fix
        _, e = np.unique(H[:, 5], True)
        v1 = self.vertices[H[e, 0]]
        v2 = self.vertices[H[H[e, 4], 0]]
        v3 = self.vertices[H[H[e, 3], 0]]
        v4 = self.vertices[H[H[H[e, 4], 3], 0]]
        be = boundary_edges(H)
        v3[be] = v1[be]
        v4[be] = v2[be]
        V1 = 3. / 8 * v1 + 3. / 8 * v2 + 1. / 8 * v3 + 1. / 8 * v4
        bh = np.where(H[:, 1] == -1)[0]
        v0 = self.vertices[H[bh, 0]]
        v5 = self.vertices[H[H[bh, 3], 0]]
        v6 = self.vertices[H[H[bh, 2], 0]]
        V0[H[bh, 0]] = 1. / 8 * v6 + 1. / 8 * v5 + 3. / 4 * v0
        V0[fixed_vertices] = v_fix
        self._update_vertices(np.vstack((V0, V1)))
        self._update_faces(new_faces)

    def catmull_clark(self, fixed_vertices=[]):
        nV = self._V
        nE = self._E
        H = self.halfedges
        h = inner_halfedges(H)
        v_fix = self.vertices[fixed_vertices]
        v1 = H[h, 0]
        v2 = H[h, 5] + nV
        v3 = H[h, 1] + nV + nE
        v4 = H[H[h, 3], 5] + nV
        h_in = inner_halfedges(H)
        h_bd = boundary_halfedges(H)
        nHi = len(h_in)
        h_map = np.zeros(len(H), dtype=int)
        h_map[h_in] = np.arange(len(h_in))
        h_map[h_bd] = np.arange(len(h_bd)) + 4*nHi
        h = np.arange(nHi)
        H1 = np.column_stack((v1, h, h + nHi, h + 3*nHi,
                              h_map[H[H[h_in, 4], 2]] + 3*nHi, h))
        H2 = np.column_stack((v2, h, h + 2*nHi, h,
                              h_map[H[h_in, 2]] + 2*nHi,
                              h + 2*nE))
        H3 = np.column_stack((v3, h, h + 3*nHi, h + nHi,
                              h_map[H[h_in, 3]] + nHi, h))
        H4 = np.column_stack((v4, h, h, h + 2*nHi,
                              h_map[H[H[h_in, 3], 4]], h))
        H_new = np.vstack((H1, H2, H3, H4))
        nHb = len(h_bd)
        v5 = H[H[h_bd, 4], 5] + nV
        v6 = H[h_bd, 0]
        f = np.repeat(-1, nHb)
        H5 = np.column_stack((v6, f, h_map[h_bd] + nHb,
                              h_map[H[h_bd, 3]] + nHb,
                              h_map[H[H[h_bd, 4], 2]] + 3*nHi,
                              f))
        H6 = np.column_stack((v5, f, h_map[H[h_bd, 2]], h_map[h_bd],
                              h_map[H[h_bd, 4]],
                              f))
        H_new[h_map[H[h_bd, 4]], 4] = h_map[h_bd] + nHb
        H_new[H_new[H_new[H_new[h_map[H[h_bd, 4]], 2], 4], 2], 4] = h_map[h_bd]
        H_new = np.vstack((H_new, H5, H6))
        e_h = edges_halfedges(H)
        e = np.arange(len(e_h))
        H_new[h_map[e_h[:, 0]], 5] = e
        H_new[H_new[h_map[e_h[:, 0]], 4], 5] = e
        H_new[h_map[e_h[:, 1]], 5] = e + nE
        H_new[H_new[h_map[e_h[:, 1]], 4], 5] = e + nE
        H_new[H_new[nHi+h, 4], 5] = H_new[nHi+h, 5]
        v = H[:, 0]
        i = np.argsort(v)
        v = v[i]
        v1 = H[H[:, 4], 0][i]
        v2 = H[H[H[:, 2], 4], 0][i]
        s = np.ones(len(v))
        li = sum_repeated(s, v)
        d = 1. / (4 * li ** 2)
        c = 3. / (2 * li ** 2)
        e = 1 - 7. / (4 * li)
        c = np.array([c, c, c]).T
        d = np.array([d, d, d]).T
        e = np.array([e, e, e]).T
        v1 = sum_repeated(self.vertices[v1], v)
        v2 = sum_repeated(self.vertices[v2], v)
        old_vertices = c * v1 + d * v2 + e * self.vertices
        old_vertices[fixed_vertices] = v_fix
        _, e = np.unique(H[:, 5], True)
        v1 = self.vertices[H[e, 0]]
        v2 = self.vertices[H[H[e, 4], 0]]
        v3 = self.vertices[H[H[e, 3], 0]]
        v4 = self.vertices[H[H[H[e, 4], 3], 0]]
        v5 = self.vertices[H[H[H[e, 2], 2], 0]]
        v6 = self.vertices[H[H[H[H[e, 4], 2], 2], 0]]
        be = boundary_edges(H)
        v3[be] = v5[be] = v1[be]
        v4[be] = v6[be] = v2[be]
        mid_points = 3. / 8 * (v1 + v2) + 1. / 16 * (v3 + v4 + v5 + v6)
        bh = np.where(H[:, 1] == -1)[0]
        v0 = self.vertices[H[bh, 0]]
        v5 = self.vertices[H[H[bh, 3], 0]]
        v6 = self.vertices[H[H[bh, 2], 0]]
        old_vertices[H[bh, 0]] = 1. / 8 * v6 + 1. / 8 * v5 + 3. / 4 * v0
        old_vertices[fixed_vertices] = v_fix
        barycenters = faces_centroid(self.vertices, self.halfedges)
        new_vertices = np.vstack((old_vertices, mid_points, barycenters))
        self._update_vertices(new_vertices)
        self._update_halfedges(H_new)

    def dual_mesh(self):
        H = self.halfedges
        dual_halfedges = np.copy(H)
        b = np.where(H[:, 1] == -1)[0]
        B = b.shape[0]
        fb = np.arange(B) + self._F
        hb1 = np.arange(B) + 2 * self._E
        hb2 = np.arange(B) + B + 2 * self._E
        eb = np.arange(B) + self._E
        dual_halfedges[b, 1] = np.copy(fb)
        dual_halfedges[b, 3] = np.copy(hb1)
        dual_halfedges[H[b, 3], 2] = np.copy(hb2)
        Hb1 = np.zeros((B, 6), dtype=np.int)
        Hb2 = np.zeros((B, 6), dtype=np.int)
        Hb1[:, 0] = -1
        Hb2[:, 0] = H[b, 0]
        Hb1[:, 1] = fb
        Hb2[:, 1] = dual_halfedges[H[b, 3], 1]
        Hb1[:, 2] = b
        Hb2[:, 2] = dual_halfedges[H[b, 3], 3]  # HD[H[b,2],2]##
        Hb1[:, 3] = dual_halfedges[b, 2]
        Hb2[:, 3] = H[b, 3]
        Hb1[:, 4] = hb2
        Hb2[:, 4] = hb1
        Hb1[:, 5] = eb
        Hb2[:, 5] = eb
        dual_halfedges = np.vstack((dual_halfedges, Hb1, Hb2))
        HR = np.copy(dual_halfedges)
        dual_halfedges[:, 0] = HR[HR[:, 4], 1]
        dual_halfedges[:, 1] = HR[:, 0]
        dual_halfedges[:, 2] = HR[HR[:, 3], 4]
        dual_halfedges[:, 3] = HR[HR[:, 4], 2]
        dual_vertices = faces_centroid(self.vertices, self.halfedges)
        new_vertices = edges_mid_point(self.vertices, self.halfedges)[H[b, 5]]
        dual_vertices = np.vstack((dual_vertices, new_vertices))
        self._update_vertices(dual_vertices)
        self._update_halfedges(dual_halfedges)

    # -------------------------------------------------------------------------
    #                             Deletion
    # -------------------------------------------------------------------------

    def delete_faces(self, face_index):
        if type(face_index) is int:
            face_index = [face_index]
        H = self.halfedges
        self._open_trash()
        self._cull_faces(face_index)
        hf = np.arange(H.shape[0])[np.in1d(H[:, 1], face_index)]
        bh = hf[H[H[hf, 4], 1] == -1]
        self._cull_halfedges(bh)
        self._cull_halfedges(H[bh, 4])
        self._cull_edges(H[bh, 5])
        #dh = hf[np.in1d(H[hf, 4], hf)]
        #self._cull_halfedges(dh)
        #self._cull_halfedges(H[dh, 4])
        #self._cull_edges(H[dh, 5])
        H[hf, 1] = -1
        H[H[H[bh, 4], 2], 3] = H[bh, 3]
        H[H[H[bh, 4], 3], 2] = H[bh, 2]
        H[H[bh, 2], 3] = H[H[bh, 4], 3]
        H[H[bh, 3], 2] = H[H[bh, 4], 2]
        self._clean()
        self.delete_unconnected_vertices()

    def delete_edges(self, edge_indices):
        if type(edge_indices) is int:
            edge_indices = [edge_indices]
        self._open_trash()
        for e in edge_indices:
            h = self.edge_halfedge(e)
            if len(self.halfedge_ring(h)) < 3:
                return False
            if len(self.halfedge_ring(self.halfedges[h, 4])) < 3:
                return False
            self._delete_halfedge(h)
        self._clean()

    def delete_unconnected_vertices(self):
        H = self.halfedges
        v = np.arange(len(self.vertices))
        cull = np.invert(np.in1d(v, H[:, 0]))
        self._open_trash()
        self._cull_vertices(v[cull])
        self._clean()

    # -------------------------------------------------------------------------
    #                               Welding
    # -------------------------------------------------------------------------

    def explode_faces(self):
        f, v = faces_vertices(self.halfedges)
        new_vertices = self.vertices[v]
        k = np.arange(len(v))
        new_faces = [[] for i in range(self._F)]
        for i in range(len(f)):
            new_faces[f[i]].append(k[i])
        self._update_vertices(new_vertices)
        self._update_faces(new_faces)

    def weld(self, tolerance=0.001):
        tree = spatial.cKDTree(self.vertices)
        k = tree.query_ball_point(self.vertices, tolerance)
        self._open_trash()
        for v in k:
            if len(v) > 1:
                for i in v:
                    self.halfedges[self.halfedges[:, 0] == i, 0] = min(v)
                    self._cull_vertex(i)
        self._clean()
        return self.vertices, self.halfedges

    def check_collapsed_vertices(self, tolerance=1e-6):
        tree = spatial.cKDTree(self.vertices)
        k = tree.query_ball_point(self.vertices, tolerance)
        for v in k:
            if len(v) > 1:
                print('*** collapsed vertices {}'.format(v))

    # -------------------------------------------------------------------------
    #                            Edit edge connectivity
    # -------------------------------------------------------------------------

    def flip_edge(self, edge_index):
        h = self.edge_halfedge(edge_index)
        if not self.is_halfedge_bounding_tri_faces(h):
            return False
        self._flip_halfedge(h)
        # self._topology_update()

    def split_edge(self, edge_index):
        h = self.edge_halfedge(edge_index)
        if not self.is_halfedge_bounding_tri_faces(h):
            return False
        self._expand_arrays()
        self._split_halfedge(h)
        self._cull_arrays()

    def collapse_edge(self, edge_index):
        h = self.edge_halfedge(edge_index)
        if not self.is_halfedge_bounding_tri_faces(h):
            return False
        self._open_trash()
        self._collapse_halfedge(h)
        self._clean()

    # -------------------------------------------------------------------------
    # REMESH
    # -------------------------------------------------------------------------

    def equalize_valences(self):
        H = self.halfedges
        _, he = np.unique(H[:, 5], return_index=True)
        t = np.repeat(6, self._V)
        t[boundary_vertices(H)] = 4
        t[mesh_corners(self.vertices, self.halfedges)] = 3
        _, _, l = vertices_ring_vertices(H, True, False, True)
        counter = 0
        for h in he:
            a = H[h, 0]
            b = H[H[h, 4], 0]
            c = H[H[h, 3], 0]
            d = H[H[H[h, 4], 3], 0]
            deviation_0 = (l[a] - t[H[h, 0]]) ** 2
            deviation_0 += (l[b] - t[H[H[h, 4], 0]]) ** 2
            deviation_0 += (l[c] - t[H[H[h, 3], 0]]) ** 2
            deviation_0 += (l[d] - t[H[H[H[h, 4], 3], 0]]) ** 2
            deviation_1 = (l[a] - t[H[h, 0]] - 1) ** 2
            deviation_1 += (l[b] - t[H[H[h, 4], 0]] - 1) ** 2
            deviation_1 += (l[c] - t[H[H[h, 3], 0]] + 1) ** 2
            deviation_1 += (l[d] - t[H[H[H[h, 4], 3], 0]] + 1) ** 2
            if deviation_1 < deviation_0:
                if self._flip_halfedge(h):
                    l[a] -= 1
                    l[b] -= 1
                    l[c] += 1
                    l[d] += 1
                    counter += 1
        self._topology_update()
        print('  flip: {} edges'.format(counter))
        return True

    def split_edges(self, max_length):
        if not self.is_triangular_mesh():
            return False
        H = self.halfedges
        _, he = np.unique(H[:, 5], return_index=True)
        self._expand_arrays(len(he))
        counter = 0
        for h in he:
            if self.halfedge_length(h) > max_length:
                if self._split_halfedge(h):
                    counter += 1
        self._cull_arrays()
        print('  split: {} edges'.format(counter))
        return True

    def collapse_edges(self, min_length):
        if not self.is_triangular_mesh():
            return False
        H = self.halfedges
        _, he = np.unique(H[:, 5], return_index=True)
        self._open_trash()
        counter = 0
        for h in he:
            if self._is_culled_halfedge(h):
                h = H[h, 4]
            if self.halfedge_length(h) < min_length:
                if self._collapse_halfedge(h):
                    counter += 1
        self._clean()
        print('  collapse: {} edges'.format(counter))
        return True

    def tangential_relaxation(self, projection_callback=None):
        b = vertices_ring_barycenter(self.vertices, self.halfedges,
                                     fixed_boundary=self.fixed_boundary)
        n = vertices_normal(self.vertices, self.halfedges)
        self.vertices = b + np.einsum('i,ij->ij', np.einsum('ij,ij->i', n, self.vertices - b), n)
        if callable(projection_callback):
            self.vertices = projection_callback(self.vertices)

    def relaxation(self, projection_callback=None):
        for i in range(3):
            b = vertices_ring_barycenter(self.vertices, self.halfedges,
                                         fixed_boundary=self.fixed_boundary)
            self.vertices += 0.3 * (b - self.vertices)
        if callable(projection_callback):
            self.vertices = projection_callback(self.vertices)

    def incremental_remesh_step(self, target_length, projection_callback=None,
                                fixed_boundary=False):
        if not self.is_triangular_mesh():
            return False
        self.fixed_boundary = fixed_boundary
        max_length = 4/3 * target_length
        min_length = 4/5 * target_length
        self.split_edges(max_length)
        self.collapse_edges(min_length)
        self.equalize_valences()
        self.tangential_relaxation(projection_callback=projection_callback)
        return True

    # -------------------------------------------------------------------------
    #                             Local connectivity
    # -------------------------------------------------------------------------

    def edge_halfedge(self, edge_index):
        H = self.halfedges
        h = np.where(H[:, 5] == edge_index)[0][0]
        return h

    def vertex_halfedge(self, vertex_index):
        H = self.halfedges
        v = np.where(H[:, 0] == vertex_index)[0][0]
        return v

    def halfedge_ring(self, halfedge_index):
        H = self.halfedges
        h0 = halfedge_index
        ring = [h0]
        h = H[H[h0, 3], 4]
        while h != h0:
            ring.append(h)
            h = H[H[h, 3], 4]
        return ring

    def vertex_ring_vertices(self, vertex_index):
        h = self.vertex_halfedge(vertex_index)
        ring = self.halfedge_ring_vertices(h)
        return ring

    def vertex_multiple_ring_vertices(self, vertex_index, depth=1):
        vi, vj = vertices_ring_vertices(self.halfedges)
        ring = np.array([], dtype='i')
        search = np.array([vertex_index], dtype='i')
        for i in range(int(depth)):
            vring = np.array([], dtype='i')
            for v in search:
                vring = np.hstack((vj[vi == v], vring))
            vring = np.unique(vring)
            vring = vring[np.invert(np.in1d(vring, ring))]
            search = vring
            ring = np.hstack((ring, vring))
            if len(ring) == self.V:
                return ring
        return np.unique(ring)

    def halfedge_ring_vertices(self, halfedge_index):
        H = self.halfedges
        ring = self.halfedge_ring(halfedge_index)
        vertices = H[H[ring, 2], 0]
        return vertices

    def halfedge_ring_faces(self, halfedge_index):
        H = self.halfedges
        ring = self.halfedge_ring(halfedge_index)
        faces = H[H[ring, 2], 1]
        return faces

    """
    def halfedge_face_vertices(self, halfedge_index):
        H = self.halfedges
        #ring = self.halfedge_face_ring(halfedge_index)
        vertices = H[ring, 0]
        return vertices
    """

    # -------------------------------------------------------------------------
    #                             Local queries
    # -------------------------------------------------------------------------

    def is_boundary_halfedge(self, halfedge_index):
        H = self.halfedges
        if H[halfedge_index, 1] < 0 or H[H[halfedge_index, 4], 1] < 0:
            return True
        else:
            return False

    def is_boundary_halfedge_ring(self, ring):
        H = self.halfedges
        for h in ring:
            if H[h, 1] < 0:
                v0 = H[h, 0]
                v1 = H[H[h, 2], 0]
                v2 = H[H[h, 3], 0]
                E1 = (self.vertices[v1] - self.vertices[v0])
                E2 = (self.vertices[v0] - self.vertices[v2])
                E1 = E1 / (E1[0] ** 2 + E1[1] ** 2 + E1[2] ** 2 + 1e-10) ** 0.5
                E2 = E2 / (E2[0] ** 2 + E2[1] ** 2 + E2[2] ** 2 + 1e-10) ** 0.5
                dot = E1[0] * E2[0] + E1[1] * E2[1] + E1[2] * E2[2]
                if dot < self.corner_tolerance:
                    return 2
                else:
                    return 1
        return 0

    def is_halfedge_bounding_tri_faces(self, halfedge_index):
        H = self.halfedges
        h = halfedge_index
        for i in range(2):
            counter = 1
            h0 = h
            h = H[h, 2]
            while h != h0:
                h = H[h, 2]
                counter += 1
                if counter > 3:
                    return False
            h = H[halfedge_index, 4]
        return True

    # -------------------------------------------------------------------------
    #                             Local geometry
    # -------------------------------------------------------------------------

    def halfedge_length(self, halfedge_index):
        H = self.halfedges
        h = halfedge_index
        V1 = self.vertices[H[h, 0]]
        V2 = self.vertices[H[H[h, 4], 0]]
        E = V1 - V2
        return (E[0] ** 2 + E[1] ** 2 + E[2] ** 2) ** 0.5

    # -------------------------------------------------------------------------
    #                         Edit half-edge connectivity
    # -------------------------------------------------------------------------

    def _delete_halfedge(self, halfedge_index):
        H = self.halfedges
        h1 = int(halfedge_index)
        h2 = H[h1, 4]
        f1 = H[h1, 1]
        f2 = H[h2, 1]
        if f1 < 0 or f2 < 0:
            return False
        h3 = H[h1, 2]
        h4 = H[h1, 3]
        h5 = H[h2, 2]
        h6 = H[h2, 3]
        H[h3, 3] = h6
        H[h6, 2] = h3
        H[h5, 3] = h4
        H[h4, 2] = h5
        H[np.where(H[:, 1] == f2)[0], 1] = f1
        self._cull_halfedge(h1)
        self._cull_halfedge(h2)
        self._cull_edge(H[h1, 5])
        self._cull_face(f2)
        self._F -= 1
        self._E -= 1
        return True

    def _flip_halfedge(self, halfedge_index):
        H = self.halfedges
        h1 = int(halfedge_index)
        h2 = H[h1, 4]
        f1 = H[h1, 1]
        f2 = H[h2, 1]
        if f1 < 0 or f2 < 0:
            return False
        ring1 = self.halfedge_ring_vertices(h1)
        ring2 = self.halfedge_ring_vertices(h2)
        if len(ring1) <= 3 or len(ring2) <= 3:
            return False
        v3 = H[H[h1, 3], 0]
        v4 = H[H[h2, 3], 0]
        ring3 = self.halfedge_ring_vertices(H[h1, 3])
        for h in ring3:
            vertices = [H[h, 0], H[H[h, 4], 0]]
            if v3 in vertices and v4 in vertices:  # ?
                return False
        h3 = H[h1, 2]
        h4 = H[h1, 3]
        h5 = H[h2, 2]
        h6 = H[h2, 3]
        i = [h1, h1, h1, h2, h2, h2, h3, h3, h4, h4, h5, h5, h6, h6, h6, h4]
        j = [0, 3, 2, 0, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1, 1]
        d = [v3, h3, h6, v4, h5, h4, h6, h1, h2, h5, h4, h2, h1, h3, f1, f2]
        H[i, j] = np.array(d)
        return True

    def _split_halfedge(self, halfedge_index):
        if self.fixed_boundary:
            if self.is_boundary_halfedge(halfedge_index):
                return False
        V = self._V
        E = self._E
        F = self._F
        K = int(2 * E)
        H = self.halfedges
        h1 = int(halfedge_index)
        h2 = H[h1, 4]
        v1 = H[h1, 0]
        v2 = H[h2, 0]
        f1 = H[h1, 1]
        f2 = H[h2, 1]
        H1 = np.copy(H[h1])
        H2 = np.copy(H[h2])
        new_vertex = (self.vertices[v1] + self.vertices[v2]) / 2
        H[h2, 0] = V
        H[h1, 2] = K
        H[h2, 3] = K + 1
        H[H2[3], 2] = K + 1
        H[H1[2], 3] = K
        nh1 = K + 0
        nh2 = K + 1
        H[K, :] = [V, f1, H1[2], h1, K + 1, E]
        H[K + 1, :] = [v2, f2, h2, H2[3], K, E]
        K0 = K + 0
        K += 2
        E += 1
        if f1 > -1:
            v3 = H[H1[3], 0]
            H[K0, 1] = F
            H[K0, 3] = K + 1
            H[H[K0, 2], 1] = F
            H[H[K0, 2], 2] = K + 1
            H[H[K0, 2], 3] = nh1
            H[H1[3], 3] = K
            H[h1, 2] = K
            H[K, :] = [V, f1, H1[3], h1, K + 1, E]
            H[K + 1, :] = [v3, F, nh1, H1[2], K, E]
            K += 2
            F += 1
            E += 1
        if f2 > -1:
            v4 = H[H2[3], 0]
            H[K0 + 1, 1] = F
            H[K0 + 1, 2] = K + 1
            H[H[K0 + 1, 3], 1] = F
            H[H[K0 + 1, 3], 3] = K + 1
            H[H[K0 + 1, 3], 2] = nh2
            H[H2[2], 2] = K
            H[h2, 3] = K
            H[K, :] = [v4, f2, h2, H2[2], K + 1, E]
            H[K + 1, :] = [V, F, H2[3], nh2, K, E]
            F += 1
            E += 1
        self.halfedges = H
        self.vertices[V, :] = new_vertex
        self._F = F
        self._E = E
        self._V += 1
        return True

    def _collapse_halfedge(self, halfedge_index):
        if self.fixed_boundary:
            if self.is_boundary_halfedge(halfedge_index):
                return False
        H = self.halfedges
        h1 = halfedge_index
        if self._is_culled_halfedge(h1):
            return False
        e = H[h1, 5]
        h2 = H[h1, 4]
        H1 = np.copy(H[h1])
        H2 = np.copy(H[h2])
        h3 = H1[2]
        h4 = H1[3]
        h5 = H2[2]
        h6 = H2[3]
        h7 = H[h3, 4]
        h8 = H[h4, 4]
        h9 = H[h5, 4]
        h10 = H[h6, 4]
        f1 = H1[1]
        f2 = H2[1]
        v1 = H[h1, 0]
        v2 = H[h2, 0]
        ring1 = self.halfedge_ring(h1)
        ring2 = self.halfedge_ring(h2)
        vring1 = H[H[ring1, 2], 0]
        vring2 = H[H[ring2, 2], 0]
        counter = 0
        for v in vring1:
            if v in vring2:
                counter += 1
        if f1 > -1 and f2 > -1 and counter > 2:
            return False
        if f1 < 0 and f2 > -1 and counter > 1:
            return False
        if f2 < 0 and f1 > -1 and counter > 1:
            return False
        V1 = np.copy(self.vertices[v1])
        V2 = np.copy(self.vertices[v2])
        bv1 = self.is_boundary_halfedge_ring(ring1)
        bv2 = self.is_boundary_halfedge_ring(ring2)
        if bv1:
            if not bv2:
                new_v = V1
            elif f1 > -1 and f2 > -1:
                return False
            elif bv1 == 2:
                new_v = V1
            elif bv2 == 2:
                new_v = V2
            else:
                new_v = (V1 + V2) / 2
        elif bv2:
            new_v = V2
        else:
            new_v = (V1 + V2) / 2
        H[ring2, 0] = v1
        self.vertices[v1] = new_v
        self._cull_halfedge(h1)
        self._cull_halfedge(h2)
        self._cull_edge(e)
        self._cull_vertex(v2)
        self._V -= 1
        self._E -= 1
        if f1 > -1:
            H[h7, 4] = h8
            H[h8, 4] = h7
            H[h7, 5] = H[h8, 5]
            self._cull_halfedge(h3)
            self._cull_halfedge(h4)
            self._cull_edge(H[h3, 5])
            self._cull_face(f1)
            self._F -= 1
            self._E -= 1
        else:
            H[h4, 2] = h3
            H[h3, 3] = h4
        if f2 > -1:
            H[h9, 4] = h10
            H[h10, 4] = h9
            H[h10, 5] = H[h9, 5]
            self._cull_halfedge(h5)
            self._cull_halfedge(h6)
            self._cull_edge(H[h6, 5])
            self._cull_face(f2)
            self._F -= 1
            self._E -= 1
        else:
            H[h6, 2] = h5
            H[h5, 3] = h6
        return True

    def _is_culled_halfedge(self, halfedge_index):
        return self._h_trash[halfedge_index]

    def _cull_halfedge(self, halfedge_index):
        self._h_trash[halfedge_index] = True

    def _cull_face(self, face_index):
        self._f_trash.append(face_index)

    def _cull_edge(self, edge_index):
        self._e_trash.append(edge_index)

    def _cull_vertex(self, vertex_index):
        self._v_trash.append(vertex_index)

    def _cull_halfedges(self, halfedge_indices):
        self._cull_halfedge(halfedge_indices)

    def _cull_faces(self, face_indices):
        try:
            face_indices = face_indices.tolist()
        except AttributeError:
            pass
        self._f_trash.extend(face_indices)

    def _cull_edges(self, edge_indices):
        try:
            edge_indices = edge_indices.tolist()
        except AttributeError:
            pass
        self._e_trash.extend(edge_indices)

    def _cull_vertices(self, vertex_indices):
        self._v_trash.extend(vertex_indices)

    def _expand_arrays(self, n=1):
        NH = -np.ones((6 * n, 6), 'i')
        NV = np.empty((n, 3))
        NV.fill(np.nan)
        self.halfedges = np.vstack((self.halfedges, NH))
        self.vertices = np.vstack((self.vertices, NV))

    def _cull_arrays(self):
        H = self.halfedges
        Hdelete = np.where(H[:, 0] == -1)[0]
        self.halfedges = np.delete(H, Hdelete, axis=0)
        V = self.vertices
        self.vertices = np.delete(V, np.where(np.isnan(V[:, 0]))[0], axis=0)
        self._topology_update()

    def _open_trash(self):
        self._h_trash = np.zeros(self.halfedges.shape[0], 'b')
        self._v_trash = []
        self._f_trash = []
        self._e_trash = []

    def _close_trash(self):
        self._h_trash = None
        self._v_trash = None
        self._f_trash = None
        self._e_trash = None

    def _cull_unused_vertices(self):
        v = np.arange(len(self.vertices))
        unused = v[np.invert(np.in1d(v, self.halfedges[:, 0]))]
        self._cull_vertices(unused)

    def clean(self):
        self._open_trash()
        self._clean()

    def _clean(self):
        H = self.halfedges
        self._cull_unused_vertices()
        if self._h_trash is None and self._v_trash is None and self._f_trash is None and self._e_trash is None:
            return False
        self._V = len(self.vertices)  # np.amax(self.halfedges[:, 0] + 1)
        self._F = np.amax(self.halfedges[:, 1] + 1)
        self._E = np.amax(self.halfedges[:, 5] + 1)
        self._h_trash = np.nonzero(self._h_trash)[0]
        self._h_trash = np.unique(self._h_trash)
        self._v_trash = np.unique(self._v_trash)
        self._f_trash = np.unique(self._f_trash)
        self._e_trash = np.unique(self._e_trash)
        self._v_trash = sorted(self._v_trash)
        self._f_trash = np.sort(self._f_trash)
        self._e_trash = np.sort(self._e_trash)
        try:
            V = max(self._V, max(self._v_trash))
        except ValueError:
            V = self._V
        try:
            F = max(self._F, np.max(self._f_trash))
        except ValueError:
            F = self._F
        try:
            E = max(self._E, np.max(self._e_trash))
        except ValueError:
            E = self._E
        if len(self._h_trash) > 0:
            roll = np.roll(self._h_trash, 1)
            roll[0] = 0
            ind = np.arange(len(self._h_trash))
            hd = np.repeat(ind, self._h_trash - roll)
            hd = np.hstack((hd, np.repeat(hd[-1] + 1, 2 * E - self._h_trash[-1])))
            h = np.arange(2 * E) - hd
            H[:, 2] = h[H[:, 2]]
            H[:, 3] = h[H[:, 3]]
            H[:, 4] = h[H[:, 4]]
            H = np.delete(H, self._h_trash, axis=0)
        if len(self._v_trash) > 0:
            roll = np.roll(self._v_trash, 1)
            roll[0] = 0
            ind = np.arange(len(self._v_trash))
            vd = np.repeat(ind, self._v_trash - roll)
            if len(vd) == 0:
                vd = 1
            else:
                vd = np.hstack((vd, np.repeat(vd[-1] + 1, V - self._v_trash[-1])))
            v = np.arange(V) - vd
            H[:, 0] = v[H[:, 0]]
        if len(self._f_trash) > 0:
            roll = np.roll(self._f_trash, 1)
            roll[0] = 0
            ind = np.arange(len(self._f_trash))
            fd = np.repeat(ind, self._f_trash - roll)
            if len(fd) == 0:
                fd = 1
            else:
                fd = np.hstack((fd, np.repeat(fd[-1] + 1, F - self._f_trash[-1])))
            f = np.arange(F) - fd
            f = np.hstack((f, [-1]))
            H[:, 1] = f[H[:, 1]]
        if len(self._e_trash) > 0:
            roll = np.roll(self._e_trash, 1)
            roll[0] = 0
            ind = np.arange(len(self._e_trash))
            ed = np.repeat(ind, self._e_trash - roll)
            if len(ed) == 0:
                ed = 1
            else:
                ed = np.hstack((ed, np.repeat(ed[-1] + 1, E - self._e_trash[-1])))
            e = np.arange(E) - ed
            H[:, 5] = e[H[:, 5]]
        self.vertices = np.delete(self.vertices, self._v_trash, axis=0)
        self.halfedges = H
        self._close_trash()
        self._topology_update()
        return True

    def _connectivity_check(self):
        H = self.halfedges
        out = '*** Mesh connectivity check ***\n'
        for h in range(len(H)):
            if H[H[h, 4], 4] != h:
                out += 'halfedge {}: twin error\n'.format(h)
            if H[H[h, 4], 5] != H[h, 5]:
                out += 'halfedge {}: edge error\n'.format(h)
            if H[H[h, 2], 1] != H[h, 1]:
                out += 'halfedge {}: next face error\n'.format(h)
            if H[H[h, 3], 1] != H[h, 1]:
                out += 'halfedge {}: previous face error\n'.format(h)
            if H[H[h, 2], 0] != H[H[h, 4], 0]:
                out += 'halfedge {}: next twin origin error\n'.format(h)
            if H[H[H[h, 3], 4], 0] != H[h, 0]:
                out += 'halfedge {}: previous twin origin error\n'.format(h)
        for v in range(self._V):
            origins = np.where(H[:, 0] == v)[0]
            if len(origins) < 2:
                out += 'vertex {}: incoming edges < 2\n'.format(v)
        for e in range(self._E):
            halfedges = np.where(H[:, 5] == e)[0]
            if len(halfedges) != 2:
                out += 'edge {}: assigned to {} halfedges\n'.format(e, len(halfedges))
        if len(out) == 32:
            out += '*** passed ***\n'
        else:
            out += '*** completed ***'
        print(out)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------


def clean_mesh(vertices, halfedges):
    mesh = Mesh(vertices, halfedges)
    mesh.clean()
    return mesh.vertices, mesh.halfedges

# -----------------------------------------------------------------------------
# Subdivision
# -----------------------------------------------------------------------------


def loop_subdivision(vertices, halfedges, fixed_vertices=[]):
    mesh = Mesh(vertices, halfedges)
    mesh.loop(fixed_vertices)
    return mesh.vertices, mesh.halfedges


def catmull_clark_subdivision(vertices, halfedges, fixed_vertices=[]):
    mesh = Mesh(vertices, halfedges)
    mesh.catmull_clark(fixed_vertices)
    return mesh.vertices, mesh.halfedges


def dual_mesh(vertices, halfedges):
    mesh = Mesh(vertices, halfedges)
    mesh.dual_mesh()
    return mesh.vertices, mesh.halfedges


# -----------------------------------------------------------------------------
# Deletion
# -----------------------------------------------------------------------------


def delete_faces(vertices, halfedges, face_indices):
    mesh = Mesh(vertices, halfedges)
    mesh.delete_faces(face_indices)
    return mesh.vertices, mesh.halfedges


def delete_edges(vertices, halfedges, edge_indices):
    mesh = Mesh(vertices, halfedges)
    mesh.delete_edges(edge_indices)
    return mesh.vertices, mesh.halfedges


def delete_unconnected_vertices(vertices, halfedges):
    mesh = Mesh(vertices, halfedges)
    mesh.delete_unconnected_vertices()
    return mesh.vertices, mesh.halfedges


# -----------------------------------------------------------------------------
# Remeshing
# -----------------------------------------------------------------------------


def collapse_edge(vertices, halfedges, edge_index):
    mesh = Mesh(vertices, halfedges)
    mesh.collapse_edge(edge_index)
    return mesh.vertices, mesh.halfedges


def collapse_threshold_edges(vertices, halfedges, min_length, fixed_boundary=False):
    mesh = Mesh(vertices, halfedges)
    mesh.fixed_boundary = fixed_boundary
    mesh.collapse_edges(min_length)
    return mesh.vertices, mesh.halfedges


def split_edge(vertices, halfedges, edge_index):
    mesh = Mesh(vertices, halfedges)
    mesh.split_edge(edge_index)
    return mesh.vertices, mesh.halfedges


def split_threshold_edges(vertices, halfedges, max_length, fixed_boundary=False):
    mesh = Mesh(vertices, halfedges)
    mesh.fixed_boundary = fixed_boundary
    mesh.split_edges(max_length)
    return mesh.vertices, mesh.halfedges


def flip_edge(vertices, halfedges, edge_index):
    mesh = Mesh(vertices, halfedges)
    mesh.flip_edge(edge_index)
    return mesh.vertices, mesh.halfedges


def equalize_valences(vertices, halfedges):
    mesh = Mesh(vertices, halfedges)
    mesh.equalize_valences()
    return mesh.vertices, mesh.halfedges


def triangular_remesh(vertices, halfedges, target_length=None, iterations=10,
                      projection_callback=None, fixed_boundary=False):
    mesh = Mesh(vertices, halfedges)
    print('*** incremental remesh ***')
    for i in range(iterations):
        print('* iteration {}'.format(i))
        mesh.incremental_remesh_step(target_length=target_length,
                                     projection_callback=projection_callback,
                                     fixed_boundary=fixed_boundary)
    mesh.collapse_edges(0)
    return mesh.vertices, mesh.halfedges


# -----------------------------------------------------------------------------
# Welding
# -----------------------------------------------------------------------------

def weld_faces(vertices, halfedges, tolerance=1e-6):
    mesh = Mesh(vertices, halfedges)
    mesh.weld(tolerance)
    return mesh.vertices, mesh.halfedges


def explode_faces(vertices, halfedges):
    mesh = Mesh(vertices, halfedges)
    mesh.explode_faces()
    return mesh.vertices, mesh.halfedges


# -----------------------------------------------------------------------------
# Triangulation
# -----------------------------------------------------------------------------


def face_triangles(halfedges):
    """Returns the face triangles ordered counterclockwise.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges
    return_face_indices : Bool
        If True it returns the face index of each triangle

    Returns
    -------
    T : np.array (nT, 3)
        array of triangles.
    face_indices (optional): np.array (nT, )
        The index of the corresponding face.
    """
    f, v = faces_vertices(halfedges)
    n = sum_repeated(np.ones(len(f)), f).astype(int)
    Fl = faces_list(halfedges)
    F = np.empty((np.max(f) + 1, int(np.max(n))))
    F.fill(np.nan)
    mask = np.arange(n.max()) < n[:, None]
    F[mask] = np.concatenate(Fl)
    T = F[:, :3]
    for i in range(n.max() - 3):
        nT = np.column_stack((F[:, 0], F[:, i+2:i+4]))
        T = np.vstack((T, nT))
    T = np.delete(T, np.where(np.isnan(T[:, 2]))[0], axis=0).astype(int)
    return T


def remesh_test(V, H, factor=0.95):
    mesh = Mesh(V, H)
    from geolab import mean_edge_length
    l = mean_edge_length(V, H)
    max_length = 4/3 * l * factor
    min_length = 4/5 * l * factor
    for i in range(3):
        print('***** iteration {} *****'.format(i))
        mesh.check_collapsed_vertices()
        print('collapse')
        mesh.collapse_edges(min_length)
        mesh.check_collapsed_vertices()
        #mesh._connectivity_check()
        print('split')
        mesh.split_edges(max_length)
        mesh.check_collapsed_vertices()
        #mesh._connectivity_check()
        print('flip')
        mesh.equalize_valences()
        mesh.check_collapsed_vertices()
        #mesh._connectivity_check()
        mesh.tangential_relaxation()

    from geolab import plotter
    p = plotter()
    p.plot_faces(mesh.vertices, halfedges=mesh.halfedges, edge_visibility=True,
                 smooth=False)
    from geolab import boundary_contiguous_vertices
    bd = boundary_contiguous_vertices(mesh.halfedges)
    for v in bd:
        p.plot_points(mesh.vertices[v], color='r', radius=0.01)
    geolab.save_mesh_obj(mesh.vertices, halfedges=mesh.halfedges, file_name='test')
    p.show()
    return mesh.vertices, mesh.halfedges


if __name__ == '__main__':
    V, F = geolab.mesh_icosahedron()
    H = geolab.halfedges(F)
    V, H = geolab.delete_faces(V, H, [0, 11])


