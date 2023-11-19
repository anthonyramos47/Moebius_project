"""
Mesh geometry functions
"""

__author__ = 'Davide Pellis'

import numpy as np

from scipy import spatial

from scipy import sparse

# -----------------------------------------------------------------------------
import geolab
from geolab.mesh.globalconnectivity import faces_edge_vertices, faces_size, \
    vertices_ring_faces, edges_faces, edges, vertices_ring_vertices, faces_vertices

from geolab.utilities.arrayutilities import sum_repeated, normalize

from geolab.mesh.boundary import are_boundary_vertices

from geolab.mesh.counting import number_of_faces


# -----------------------------------------------------------------------------
#                                 Closeness
# -----------------------------------------------------------------------------


def closest_vertices(vertices, points):
    """Returns the indices of the closest vertex of a set of points.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    points : np.array (n, 3)
        An array containing a set of points where the coordinates of the i-th
        point are in the i-th row.

    Returns
    -------
    np.array (n,)
        An array that contains in i-th position the index of the closest vertex
        to the i-th point.
    """
    kd_tree = spatial.cKDTree(vertices)
    closest = kd_tree.query(points)[1]
    return closest


def snap_vertices(vertices, threshold=1e-3):
    new_vertices = np.copy(vertices)
    kd_tree = spatial.cKDTree(vertices)
    closest = kd_tree.query_ball_point(vertices, threshold)
    for snap in closest:
        if len(snap) > 1:
            new_vertices[snap] = np.sum(vertices[snap], axis=0) / len(snap)
    return new_vertices


# -----------------------------------------------------------------------------
#                                FEATURES
# -----------------------------------------------------------------------------

def feature_edges(vertices, halfedges, angle=2*np.pi/3):
    theta = edges_dihedral_angle(vertices, halfedges)
    return np.where(theta >= np.pi - angle)[0]


def edges_dihedral_angle(vertices, halfedges):
    Nf = faces_normal(vertices, halfedges)
    f = edges_faces(halfedges)
    cos = np.einsum('ij,ij->i', Nf[f[:, 0]], Nf[f[:, 1]])
    return np.arccos(cos)


# -----------------------------------------------------------------------------
#                                  Ring
# -----------------------------------------------------------------------------

def vertices_ring_barycenter(vertices, halfedges, fixed_boundary=True):
    vi, vj, l = vertices_ring_vertices(halfedges, return_lengths=True)
    B = np.einsum('ij,i->ij', sum_repeated(vertices[vj], vi), 1/l)
    if fixed_boundary:
        boundary = are_boundary_vertices(halfedges)
        B[boundary] = vertices[boundary]
    else:
        h = geolab.boundary_halfedges(halfedges)
        prev = geolab.navigate_halfedges(halfedges, 'p-o')
        nex = geolab.navigate_halfedges(halfedges, 'n-o')
        B[halfedges[h, 0]] = (vertices[prev[h]] + vertices[nex[h]]) / 2
    return B

# -----------------------------------------------------------------------------
#                                Normals
# -----------------------------------------------------------------------------


def faces_vector_area(vertices, halfedges):
    """Returns a vector normal to each face with length equal to the face area.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (F, 3)
        An array that contains in i-th position the vector area of i-yh face
        to the i-th point.
    """
    f, v1, v2 = faces_edge_vertices(halfedges, order=True)
    V1 = vertices[v1, :]
    V2 = vertices[v2, :]
    N = np.cross(V1, V2)
    normals = 0.5 * sum_repeated(N, f)
    return normals


def faces_normal(vertices, halfedges):
    """Returns the normals of faces.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (F, 3)
        An array that contains in i-th position the vector area of i-th face
        to the i-th point.
    """
    N = faces_vector_area(vertices, halfedges)
    N = normalize(N)
    return N


def vertices_normal(vertices, halfedges):
    """Returns the normals at vertices as sum of vector areas of incident
    faces.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (V, 3)
        An array that contains in i-th position the normal at i-th vertex.
    """
    N = faces_vector_area(vertices, halfedges)
    v, fi = vertices_ring_faces(halfedges, sort=True)
    N = N[fi, :]
    normals = normalize(sum_repeated(N, v))
    return normals


def edges_normal(vertices, halfedges):
    """Returns the normals at edges as normalized sum on the normals of
     incident faces

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (V, 3)
        An array that contains in i-th position the normal at i-th edge.
    """
    N = faces_normal(vertices, halfedges)
    N = np.insert(N, N.shape[0], 0, axis=0)
    f = edges_faces(halfedges)
    normals = normalize(N[f[:, 0]] + N[f[:, 1]])
    return normals


# def boundary_normals(self):
#     H = self.halfedges
#     b = np.where(H[:, 1] == -1)[0]
#     face_normals = self.face_normals()
#     N1 = face_normals[H[H[b, 4], 1]]
#     N2 = face_normals[H[H[H[b, 3], 4], 1]]
#     normals = np.zeros((self.V, 3))
#     E1 = self.vertices[H[H[b, 2], 0]] - self.vertices[H[b, 0]]
#     E2 = self.vertices[H[b, 0]] - self.vertices[H[H[b, 3], 0]]
#     N = np.cross(N1, E1) + np.cross(N2, E2)
#     N = N / np.linalg.norm(N, axis=1, keepdims=True)
#     normals[H[b, 0], :] = N
#     return normals
#
#
# def boundary_tangents(self, normalize=True):
#     H = self.halfedges
#     b = np.where(H[:, 1] == -1)[0]
#     V1 = self.vertices[H[H[b, 3], 0]]
#     V2 = self.vertices[H[H[b, 2], 0]]
#     T = (V2 - V1)
#     if normalize:
#         T = T / np.linalg.norm(T, keepdims=True)
#     else:
#         T = T / 2
#     tangents = np.zeros((self.V, 3))
#     tangents[H[b, 0], :] = T
#     return tangents


# -----------------------------------------------------------------------------
#                                  Areas
# -----------------------------------------------------------------------------


def faces_centroid(vertices, halfedges):
    """Returns the indices of the closest vertex of a set of points.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (F, 3)
        An array that contains in i-th position the coordinates of the area
        centroid of i-th face.
    """
    H = halfedges[np.where(halfedges[:, 1] >= 0)[0], :]
    i = np.argsort(H[:, 1])
    f = H[i, 1]
    V = vertices[H[i, 0], :]
    C = np.einsum('i,ij->ij', 1/faces_size(halfedges), sum_repeated(V, f))
    return C


def faces_area(vertices, halfedges):
    """Returns the area of each face (faces are implicitly triangulated).

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (E,)
        An array that contains in i-th position the area of i-th edge.
    """
    N = faces_vector_area(vertices, halfedges)
    A = np.linalg.norm(N, axis=1)
    return A


def mesh_area(vertices, halfedges):
    """Returns the area of the mesh, computed as sum of faces areas.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    float
        The area of the mesh.
    """
    A = faces_area(vertices, halfedges)
    A = np.sum(A)
    return A


def vertices_ring_area(vertices, halfedges):
    """Returns the area of each vertex-ring, computed as

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (V,)
        An array that contains in i-th position the area i-th vertex-ring.
    """
    L = faces_size(halfedges)
    Af = faces_area(vertices, halfedges)
    v, fi = vertices_ring_faces(halfedges, sort=True)
    Ar = sum_repeated(Af[fi] / L[fi], v)
    return Ar


def edges_length(vertices, halfedges):
    """Returns the length of each edge.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (V,)
        The lenght of each edge.
    """
    e = edges(halfedges)
    e = vertices[e[:, 0]] - vertices[e[:, 1]]
    return np.linalg.norm(e, axis=1)


def mean_edge_length(vertices, halfedges):
    """Returns the arithmetic mean of the edge lengths.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    float
        Mean edge length.
    """
    L = edges_length(vertices, halfedges)
    return np.sum(L) / len(L)


def edges_mid_point(vertices, halfedges):
    """Returns the mid point of each edge.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (n, 6)
        The array of halfedges.

    Returns
    -------
    np.array (E, 3)
        Edges mid points.
    """
    e = edges(halfedges)
    mid = (vertices[e[:, 0]] + vertices[e[:, 1]]) / 2
    return mid


def faces_planarity(vertices, halfedges, scale_invariant=True):
    planarity = np.zeros(number_of_faces(halfedges))
    f, vi = faces_vertices(halfedges)
    i = np.ones((f.shape[0]), dtype=np.int)
    j = np.arange(f.shape[0])
    _, k = np.unique(f, True)
    L = sum_repeated(i, f)
    index = j[k]
    quad = np.where(L > 3)[0]
    shift = 0
    while len(quad) > 0:
        P1 = vertices[vi[index[quad] + shift]]
        P2 = vertices[vi[index[quad] + shift + 1]]
        P3 = vertices[vi[index[quad] + shift + 2]]
        P4 = vertices[vi[index[quad] + shift + 3]]
        V1 = P3 - P1
        V2 = P4 - P2
        N = np.cross(V1, V2)
        eps = np.finfo(float).eps
        P12 = P2 - P1
        norm = np.linalg.norm(N, axis=1) + eps
        d = np.einsum('ij,ij->i', P12, N) / norm
        if scale_invariant:
            d1 = np.linalg.norm(V1, axis=1)
            d2 = np.linalg.norm(V1, axis=1)
            p = np.abs(d) / ((d1 + d2) / 2)
        else:
            p = np.abs(d)
        planarity[quad] = np.maximum(p, planarity[quad])
        L -= 1
        shift += 1
        quad = np.where(L > 3)[0]
    return planarity
