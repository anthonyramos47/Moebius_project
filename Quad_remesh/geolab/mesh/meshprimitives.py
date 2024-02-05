#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

# -----------------------------------------------------------------------------

from geolab.mesh.halfedges import halfedges

from geolab.geometry.polyline import polyline_vertex_frame, polyline_edges_length

from geolab.plot.combnormals import comb_normals

from geolab.transform.reframe import mesh_rigid_transformation

from geolab.geometry.frame import make_frame

from geolab.mesh.editconnectivity import join_meshes

# -----------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
#                                PRIMITIVES
# -----------------------------------------------------------------------------


def mesh_plane(x_faces=10, y_faces=10, x_range=(-5, 5), y_range=(-5, 5)):
    x_faces += 1
    y_faces += 1
    M = np.arange(x_faces * y_faces, dtype=np.int)
    M = np.reshape(M, (x_faces, y_faces))
    quads = np.zeros((x_faces - 1, y_faces - 1, 4), dtype=np.int)
    quads[:, :, 3] = M[:M.shape[0] - 1, :M.shape[1] - 1]
    quads[:, :, 2] = M[:M.shape[0] - 1, 1:]
    quads[:, :, 1] = M[1:, 1:]
    quads[:, :, 0] = M[1:, :M.shape[1] - 1]
    quads = quads.reshape(((x_faces - 1) * (y_faces - 1), 4))
    x = np.linspace(x_range[0], x_range[1], x_faces)
    y = np.linspace(y_range[0], y_range[1], y_faces)
    Px, Py = np.meshgrid(x, y)
    Px = np.reshape(Px, (x_faces * y_faces), order='F')
    Py = np.reshape(Py, (x_faces * y_faces), order='F')
    vertices = np.vstack((Px, Py, np.zeros(x_faces * y_faces))).T
    return vertices, quads


def mesh_cylinder(vertical_faces=5, around_faces=8, center=(0, 0, 0), radius=1, height=3):
    C = np.array(center)
    Fx = around_faces
    Fy = vertical_faces + 1
    M = np.arange(Fx * Fy, dtype=np.int)
    M = np.reshape(M, (Fy, Fx))
    M = np.insert(M, Fx, M[:, 0], axis=1)
    Q = np.zeros((Fy - 1, Fx, 4), dtype=np.int)
    Q[:, :, 0] = M[:M.shape[0] - 1, :M.shape[1] - 1]
    Q[:, :, 1] = M[:M.shape[0] - 1, 1:]
    Q[:, :, 2] = M[1:, 1:]
    Q[:, :, 3] = M[1:, :M.shape[1] - 1]
    faces = Q.reshape((Fx * (Fy - 1), 4), order='C')
    phi = np.linspace(0, 2 * np.pi, Fx + 1)
    phi = np.delete(phi, -1)
    z = np.linspace(C[2], C[2] + height, Fy)
    phi, z = np.meshgrid(phi, z)
    Px = radius * np.cos(phi) + C[0]
    Py = radius * np.sin(phi) + C[1]
    Px = np.reshape(Px, (Fx * Fy), order='C')
    Py = np.reshape(Py, (Fx * Fy), order='C')
    Pz = np.reshape(z, (Fx * Fy), order='C')
    vertices = np.vstack((Px, Py, Pz)).T
    return vertices, faces


def mesh_sphere(around_faces=20, vertical_faces=10, center=(0, 0, 0), radius=1):
    center = np.array(center)
    Fx = around_faces
    Fy = vertical_faces - 1
    M = np.arange(Fx * Fy, dtype=np.int)
    M = np.reshape(M, (Fy, Fx)) + 1
    M = np.insert(M, Fx, M[:, 0], axis=1)
    Q = np.zeros((Fy - 1, Fx, 4), dtype=np.int)
    Q[:, :, 3] = M[:M.shape[0] - 1, :M.shape[1] - 1]
    Q[:, :, 2] = M[:M.shape[0] - 1, 1:]
    Q[:, :, 1] = M[1:, 1:]
    Q[:, :, 0] = M[1:, :M.shape[1] - 1]
    Q = Q.reshape((Fx * (Fy - 1), 4), order='C').tolist()
    T = np.zeros((2, Fx, 3), dtype=np.int)
    T[0, :, 0] = M[0, :M.shape[1] - 1]
    T[0, :, 1] = M[0, 1:]
    T[0, :, 2] = 0
    T[1, :, 0] = M[-1, 1:]
    T[1, :, 1] = M[-1, :M.shape[1] - 1]
    T[1, :, 2] = Fx * Fy + 1
    T = T.reshape((Fx * 2, 3), order='C').tolist()
    F = Q + T
    phi = np.linspace(0, 2 * np.pi, Fx + 1)[:-1]
    theta = np.linspace(0, np.pi, Fy + 2)
    theta = theta[1:theta.shape[0] - 1]
    phi, theta = np.meshgrid(phi, theta)
    Px = radius * np.cos(phi) * np.sin(theta) + center[0]
    Py = radius * np.sin(phi) * np.sin(theta) + center[1]
    Pz = radius * np.cos(theta) + center[2]
    Px = np.reshape(Px, (Fx * Fy), order='C')
    Py = np.reshape(Py, (Fx * Fy), order='C')
    Pz = np.reshape(Pz, (Fx * Fy), order='C')
    P = np.vstack((Px, Py, Pz)).T
    P = np.insert(P, 0, [center[0], center[1], center[2] + radius], axis=0)
    P = np.insert(P, P.shape[0], [center[0], center[1], center[2] - radius], axis=0)
    H = halfedges(F)
    return P, H


def mesh_torus(ring_faces=25, section_faces=8, center=(0, 0, 0), ring_radius=3,
               section_radius=1):
    center = np.array(center)
    Fx = section_faces
    Fy = ring_faces
    M = np.arange(Fx * Fy, dtype=np.int)
    M = np.reshape(M, (Fy, Fx))
    M = np.insert(M, Fx, M[:, 0], axis=1)
    M = np.insert(M, Fy, M[0, :], axis=0)
    faces = np.zeros((Fy, Fx, 4), dtype=np.int)
    faces[:, :, 3] = M[:M.shape[0] - 1, :M.shape[1] - 1]
    faces[:, :, 2] = M[:M.shape[0] - 1, 1:]
    faces[:, :, 1] = M[1:, 1:]
    faces[:, :, 0] = M[1:, :M.shape[1] - 1]
    faces = faces.reshape((Fx * Fy, 4), order='C')
    u = np.linspace(0, 2 * np.pi, Fx + 1)[:-1]
    v = np.linspace(0, 2 * np.pi, Fy + 1)[:-1]
    U, V = np.meshgrid(u, v)
    vertices = np.array([section_radius * np.cos(U) * np.cos(V) + ring_radius * np.cos(V) + center[0],
                         section_radius * np.cos(U) * np.sin(V) + ring_radius * np.sin(V) + center[1],
                         section_radius * np.sin(U) + center[2]])
    Px = np.reshape(vertices[0], (Fx * Fy), order='C')
    Py = np.reshape(vertices[1], (Fx * Fy), order='C')
    Pz = np.reshape(vertices[2], (Fx * Fy), order='C')
    vertices = np.vstack((Px, Py, Pz)).T
    return vertices, faces


def mesh_icosahedron(center=(0, 0, 0), radius=1):
    vertices = np.array([[0, 0, 1],
                         [0.89442718029022217, 0, 0.44721359014511108],
                         [0.27639320492744446, 0.85065078735351563, 0.44721359014511108],
                         [-0.72360676527023315, 0.52573108673095703, 0.44721359014511108],
                         [-0.72360676527023315, -0.52573108673095703, 0.44721359014511108],
                         [0.27639320492744446, -0.85065078735351563, 0.44721359014511108],
                         [0.72360676527023315, 0.52573108673095703, -0.44721359014511108],
                         [-0.27639320492744446, 0.85065078735351563, -0.44721359014511108],
                         [-0.89442718029022217, 0, -0.44721359014511108],
                         [-0.27639320492744446, -0.85065078735351563, -0.44721359014511108],
                         [0.72360676527023315, -0.52573108673095703, -0.44721359014511108],
                         [0, 0, -1]])

    faces = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 2],
                      [2, 6, 11], [2, 7, 3], [2, 11, 7], [3, 7, 8], [3, 8, 4],
                      [4, 8, 9], [4, 9, 5], [5, 9, 10], [5, 10, 6], [6, 10, 11],
                      [7, 11, 12], [7, 12, 8], [8, 12, 9], [9, 12, 10],
                      [10, 12, 11]]) - 1

    vertices *= radius
    vertices[:, 0] += center[0]
    vertices[:, 1] += center[1]
    vertices[:, 2] += center[2]
    return vertices, faces


def mesh_rectangular_pipe(vertices, normals, height=0.1, width=0.05,
                          offset=0, closed=False, make_texture=False,
                          return_bottom_faces=False, normals_mode='vertex',
                          straight_end=True):
    Fx = 4
    Fy = len(vertices)

    N = normals / (np.linalg.norm(normals, axis=1, keepdims=True))
    if len(N) == len(vertices) - 1:
        N = np.vstack((N, N[-1]))
    if not closed:
        if straight_end:
            N[-1] = N[-2]
    if not closed:
        M = np.arange(Fx * Fy, dtype=np.int)
        M = np.reshape(M, (Fy, Fx))
        M = np.insert(M, Fx, M[:, 0], axis=1)
        Q = np.zeros((Fy - 1, Fx, 4), dtype=np.int)
        Q[:, :, 0] = M[:M.shape[0] - 1, :M.shape[1] - 1]
        Q[:, :, 1] = M[:M.shape[0] - 1, 1:]
        Q[:, :, 2] = M[1:, 1:]
        Q[:, :, 3] = M[1:, :M.shape[1] - 1]
        Q = Q.reshape((Fx * (Fy - 1), 4), order='C')
    else:
        M = np.arange(Fx * Fy, dtype=np.int)
        M = np.reshape(M, (Fy, Fx))
        M = np.insert(M, Fx, M[:, 0], axis=1)
        M = np.insert(M, Fy, M[0, :], axis=0)
        Q = np.zeros((Fy, Fx, 4), dtype=np.int)
        Q[:, :, 0] = M[:M.shape[0] - 1, :M.shape[1] - 1]
        Q[:, :, 1] = M[:M.shape[0] - 1, 1:]
        Q[:, :, 2] = M[1:, 1:]
        Q[:, :, 3] = M[1:, :M.shape[1] - 1]
        Q = Q.reshape((Fx * Fy, 4), order='C')
    V1, V2, V3 = polyline_vertex_frame(vertices, closed)
    if not straight_end:
        V1[0] = np.cross(V2[0], N[0])
        V1[-1] = np.cross(V2[-1], N[-1])
    E = np.roll(vertices, -1, axis=0) - vertices
    E = E / (np.linalg.norm(E, axis=1, keepdims=True))
    if not closed:
        E[-1] = E[-2]
    B = np.cross(N, E)
    P1 = vertices + (height / 2 + offset) * N + width / 2 * B
    P2 = vertices + (height / 2 + offset) * N - width / 2 * B
    P3 = vertices - (height / 2 - offset) * N - width / 2 * B
    P4 = vertices - (height / 2 - offset) * N + width / 2 * B
    L = np.einsum('ij,ij->i', E, V1)
    d1 = np.einsum('ij,ij->i', vertices - P1, V1) / L
    d2 = np.einsum('ij,ij->i', vertices - P2, V1) / L
    d3 = np.einsum('ij,ij->i', vertices - P3, V1) / L
    d4 = np.einsum('ij,ij->i', vertices - P4, V1) / L
    P1 += np.einsum('i,ij->ij', d1, E)
    P2 += np.einsum('i,ij->ij', d2, E)
    P3 += np.einsum('i,ij->ij', d3, E)
    P4 += np.einsum('i,ij->ij', d4, E)
    V = np.hstack((P1, P2, P3, P4))
    V = V.reshape((4 * len(vertices), 3), order='C')
    if make_texture:
        # TODO
        Lu = np.hstack(([0], polyline_edges_length(vertices, closed)))
        Lv = np.array([0, width, width + height, height])
        u = np.repeat(np.cumsum(Lu), 4)
        v = np.tile(Lv, len(vertices))
        UV = np.column_stack((u, v))
        UV /= np.max(UV)
        return V, Q, UV
    if return_bottom_faces:
        BF = np.arange(2, len(V) - 4, 4)
        return V, Q, BF
    return V, Q


def mesh_circular_pipe(vertices, radius=0.1, closed=False, sides=48, comb=False,
                       vertex_normals=None):
    Fx = sides
    Fy = len(vertices)
    if not closed:
        M = np.arange(Fx * Fy, dtype=np.int)
        M = np.reshape(M, (Fy, Fx))
        M = np.insert(M, Fx, M[:, 0], axis=1)
        Q = np.zeros((Fy - 1, Fx, 4), dtype=np.int)
        Q[:, :, 0] = M[:M.shape[0] - 1, :M.shape[1] - 1]
        Q[:, :, 1] = M[:M.shape[0] - 1, 1:]
        Q[:, :, 2] = M[1:, 1:]
        Q[:, :, 3] = M[1:, :M.shape[1] - 1]
        Q = Q.reshape((Fx * (Fy - 1), 4), order='C')
    else:
        M = np.arange(Fx * Fy, dtype=np.int)
        M = np.reshape(M, (Fy, Fx))
        M = np.insert(M, Fx, M[:, 0], axis=1)
        M = np.insert(M, Fy, M[0, :], axis=0)
        Q = np.zeros((Fy, Fx, 4), dtype=np.int)
        Q[:, :, 0] = M[:M.shape[0] - 1, :M.shape[1] - 1]
        Q[:, :, 1] = M[:M.shape[0] - 1, 1:]
        Q[:, :, 2] = M[1:, 1:]
        Q[:, :, 3] = M[1:, :M.shape[1] - 1]
        Q = Q.reshape((Fx * Fy, 4), order='C')
    V1, N, V3, C = polyline_vertex_frame(vertices, closed, return_cosine=True)
    if vertex_normals is not None:
        if len(vertex_normals) == len(vertices):
            N = vertex_normals
    if comb:
        V2 = comb_normals(vertices, closed)
        V3 = np.cross(V1, V2)
        sin = np.einsum('ij,ij->i', N, V2)
        cos = np.einsum('ij,ij->i', N, V3)
        alpha = np.einsum('ij,ij->i', np.cross(N, V2), V3)
        alpha = np.arcsin(alpha)
        alpha[sin < 0] += np.pi
        B = np.sqrt(np.abs(0.5 * (1 + C))) + 1e-10
        B = radius / B - radius
        B = np.hstack((B, B, B))
        phi = np.linspace(0, 2 * np.pi, Fx + 1)
        phi = np.delete(phi, -1)
        r = radius
        i = np.repeat(np.arange(Fy), Fx)
        phi = np.tile(phi, Fy)
        Vi = vertices
        alpha = np.repeat(alpha, Fx)
        sin = np.repeat(sin, Fx)
        cos = np.repeat(cos, Fx)
        Px = (r * np.cos(phi) * V2[i, 0] + r * np.sin(phi) * V3[i, 0] + Vi[i, 0]
              + B[i] * sin * np.cos(phi - alpha) * V2[i, 0]
              + B[i] * cos * np.cos(phi - alpha) * V3[i, 0])
        Py = (r * np.cos(phi) * V2[i, 1] + r * np.sin(phi) * V3[i, 1] + Vi[i, 1]
              + B[i] * sin * np.cos(phi - alpha) * V2[i, 1]
              + B[i] * cos * np.cos(phi - alpha) * V3[i, 1])
        Pz = (r * np.cos(phi) * V2[i, 2] + r * np.sin(phi) * V3[i, 2] + Vi[i, 2]
              + B[i] * sin * np.cos(phi - alpha) * V2[i, 2]
              + B[i] * cos * np.cos(phi - alpha) * V3[i, 2])
    else:
        V2 = N
        B = np.sqrt(np.abs(0.5 * (1 + C))) + 1e-10
        B = radius / B
        B = np.hstack((B, B, B))
        phi = np.linspace(0, 2 * np.pi, Fx + 1)
        phi = np.delete(phi, -1)
        r = radius
        i = np.repeat(np.arange(Fy), Fx)
        phi = np.tile(phi, Fy)
        Vi = vertices
        Px = B[i] * np.cos(phi) * V2[i, 0] + r * np.sin(phi) * V3[i, 0] + Vi[i, 0]
        Py = B[i] * np.cos(phi) * V2[i, 1] + r * np.sin(phi) * V3[i, 1] + Vi[i, 1]
        Pz = B[i] * np.cos(phi) * V2[i, 2] + r * np.sin(phi) * V3[i, 2] + Vi[i, 2]
    Px = np.reshape(Px, (Fx * Fy), order='C')
    Py = np.reshape(Py, (Fx * Fy), order='C')
    Pz = np.reshape(Pz, (Fx * Fy), order='C')
    points = np.vstack((Px, Py, Pz)).T
    return points, Q


def mesh_triangular_grid(x_faces=10, y_faces=10, origin=(0, 0, 0), side=1):
    N1 = x_faces + 1
    M1 = int(y_faces / 2 + 1)
    N2 = x_faces
    M2 = int((y_faces - 1) / 2) + 1
    V1 = np.arange(N1 * M1)
    V2 = np.arange(N2 * M2)
    V1 = V1.reshape((M1, N1))
    V2 = V2.reshape((M2, N2))
    V1 += np.repeat(np.arange(M1)*N2, N1).reshape((M1, N1))
    V2 += np.repeat(np.arange(M2)*N1 + N1, N2).reshape((M2, N2))
    offset = M1 - M2
    if offset == 1:
        F1 = np.dstack((V1[:-1, 1:], V1[:-1, :-1], V2))
    else:
        F1 = np.dstack((V1[:, 1:], V1[:, :-1], V2))
    F1 = F1.reshape((N2 * (M1 - offset), 3))
    if offset == 1:
        F2 = np.dstack((V1[:-1, 1:-1], V2[:, :-1], V2[:, 1:]))
    else:
        F2 = np.dstack((V1[:, 1:-1], V2[:, :-1], V2[:, 1:]))
    F2 = F2.reshape(((N2-1) * (M1 - offset), 3))
    if offset == 1:
        F3 = np.dstack((V1[1:, :-1], V1[1:, 1:], V2))
    else:
        F3 = np.dstack((V1[1:, :-1], V1[1:, 1:], V2[:-1]))
    F3 = F3.reshape((N2 * (M1 - 1), 3))
    if offset == 1:
        F4 = np.dstack((V1[1:, 1:-1], V2[:, 1:], V2[:, :-1]))
    else:
        F4 = np.dstack((V1[1:, 1:-1], V2[:-1, 1:], V2[:-1, :-1]))
    F4 = F4.reshape(((N2 - 1) * (M1 - 1), 3))
    F = np.vstack((F1, F2, F3, F4))
    X1 = np.tile(np.arange(N1) * side, M1).reshape(M1, N1)
    Y1 = np.repeat(np.arange(M1) * side * 3**.5, N1).reshape(M1, N1)
    X2 = np.tile(np.arange(N2) * side + side/2, M2).reshape(M2, N2)
    Y2 = np.repeat(np.arange(M2) * side * 3**.5 + side * 3**.5 / 2, N2).reshape(M2, N2)
    if offset == 1:
        X2 = np.vstack((X2, np.tile(np.nan, N2)))
        Y2 = np.vstack((Y2, np.tile(np.nan, N2)))
    X = np.column_stack((X1, X2)).flatten()
    Y = np.column_stack((Y1, Y2)).flatten()
    X = X[np.logical_not(np.isnan(X))]
    Y = Y[np.logical_not(np.isnan(Y))]
    V = np.column_stack((X, -Y + np.max(Y), np.zeros(len(X))))
    return V, F


def dashed_lines(V1, V2, dash=0.1, radius=0.02):
    d = np.linalg.norm(V1 - V2, axis=1)
    n = np.round(d / (2*dash)).astype('i')
    D = (V2 - V1) / np.linalg.norm(V1 - V2, axis=1, keepdims=True)
    P1 = np.empty((0, 3))
    N1 = np.empty((0, 3))
    P0 = np.empty((0, 3))
    N0 = np.empty((0, 3))
    for i in range(len(V1)):
        x = np.linspace(V1[i, 0], V2[i, 0], n[i])
        y = np.linspace(V1[i, 1], V2[i, 1], n[i])
        z = np.linspace(V1[i, 2], V2[i, 2], n[i])
        V = np.column_stack((x, y, z))
        if len(V) == 0:
            V = np.vstack((V1, (V1 + V2) / 2, V2))
        P1 = np.vstack((P1, V[1:-1]))
        N1 = np.vstack((N1, np.tile(D[i], (len(V)-2, 1))))
        P0 = np.vstack((P0, V[0] + D[i]*dash/4))
        N0 = np.vstack((N0, D[i]))
        P0 = np.vstack((P0, V[-1] - D[i]*dash/4))
        N0 = np.vstack((N0, D[i]))
    V1, F1 = mesh_cylinder(1, 36, radius=radius, height=dash)
    V1[:, 2] -= dash/2
    R1 = make_frame(origin=P1, e3=N1)
    V, F = mesh_rigid_transformation(V1, R1, F1)
    if not np.isnan(P0[0, 0]):
        V0, F0 = mesh_cylinder(1, 36, radius=radius, height=dash/2)
        V0[:, 2] -= dash/4
        R0 = make_frame(origin=P0, e3=N0)
        V0, F0 = mesh_rigid_transformation(V0, R0, F0)
        V, F = join_meshes((V0, V), (F0, F))
    return V, F


if __name__ == '__main__':
    import geolab as geo
    V, F = mesh_triangular_grid(10, 10)
    H = geo.halfedges(F)
    e = geo.edges(H)
    V, F = dashed_lines(V[e[:, 0]], V[e[:, 1]], 0.06, 0.01)
    plotter = geo.plotter()
    plotter.plot_faces(V, F, color='k', shading=False)
    plotter.show()

