"""
Global connectivity functions
"""

__author__ = 'Davide Pellis'

import numpy as np

from scipy.sparse import coo_matrix

# -----------------------------------------------------------------------------

from geolab.utilities.arrayutilities import sum_repeated

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#                                Cell sum
# -----------------------------------------------------------------------------


def vertex_cells_sum(values, cells):
    """
    Sums vertex values from each incident n-cell.

    Parameters
    ----------
    values : np.array (#cells,) or (n #cells,)
        The cell values to be summed at vertices.
        If shape (#cells,): The value for each i-th face in i-th position.
        If shape (n #cells,): The value for the three vertices of the i-th cell
        ordered as: [v1_i, ..., vn_i].

    cells : np.array (#cells, n)
        The cells of vertices.

    Returns
    -------
    v_sum : np.array (V,)
        The sum at each i-th vertex in i-th position.
    """
    i = cells.flatten('F')
    j = np.arange(len(cells))
    j = np.tile(j, cells.shape[1])
    if len(values) == len(cells):
        values = values[j]
    v_sum = coo_matrix((values, (i, j)), (np.max(cells) + 1, len(cells)))
    return np.array(v_sum.sum(axis=1)).flatten()


# -----------------------------------------------------------------------------
#                                Connectivity
# -----------------------------------------------------------------------------


def vertices_ring_ordered_halfedges(halfedges):
    """Returns the halfedges ordered counterclockwise around each vertex.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array (H, )
        halfedge indices ordered counterclockwise around each vertex
    """
    H1 = np.copy(halfedges)
    i = np.argsort(H1[:, 0])
    v = H1[i, 0]
    index = np.arange(H1.shape[0])
    _, j = np.unique(v, True)
    v = np.delete(v, j)
    index = np.delete(index, j)
    while v.shape[0] > 0:
        _, j = np.unique(v, True)
        i[index[j]] = H1[H1[i[index[j] - 1], 3], 4]
        v = np.delete(v, j)
        index = np.delete(index, j)
    return i


def faces_ordered_halfedges(halfedges):
    """Returns the halfedges ordered counterclockwise around each face.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array (H, 6)
        halfedges indices ordered counterclockwise around each face.
    """
    H1 = np.copy(halfedges)
    i = np.argsort(H1[:, 1])
    i = i[np.where(H1[i, 1] >= 0)]
    f = H1[i, 1]
    index = np.arange(i.shape[0])
    _, j = np.unique(f, True)
    f = np.delete(f, j)
    index = np.delete(index, j)
    while f.shape[0] > 0:
        _, j = np.unique(f, True)
        i[index[j]] = H1[i[index[j] - 1], 2]
        f = np.delete(f, j)
        index = np.delete(index, j)
    return i


def vertices_ring_vertices(halfedges, sort=False, order=False, return_lengths=False):
    """Returns the vertex-ring (connected vertices) of each vertex, as
    arrays of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.
    sort : bool (optional)
        If `True`, the central vertices' indices are sorted increasingly.
        If `False` (default), the indices of the central vertex are not
        sorted but the function runs faster.
    order : bool (optional)
        If `True`, the vertex-rings are ordered contiguously
        counterclockwise around the central vertices.
        If `False` (default), the rings are not ordered contiguously but
        the function runs faster.
    return_lengths : bool (optional)
        if `True`, the function returns also the number of vertices in each
        vertex-ring (default is `False`).

    Returns
    -------
    vi : np.array
        The indices of the central vertices. Each index appears as many
        times as the number of incident edges at the vertex (vertex valence).
    vj : np.array
        The indices of the corresponding ring-vertices.
    li (optional) : np.array
        the number of vertices in each vertex-ring, sorted as v.
    """
    if order:
        i = vertices_ring_ordered_halfedges(halfedges)
        vi = halfedges[i, 0]
        vj = halfedges[halfedges[i, 4], 0]
    elif sort:
        i = np.argsort(halfedges[:, 0])
        vi = halfedges[i, 0]
        vj = halfedges[halfedges[i, 4], 0]
    else:
        vi = halfedges[:, 0]
        vj = halfedges[halfedges[:, 4], 0]
    if return_lengths:
        i = np.ones(vj.shape[0], dtype='i')
        li = sum_repeated(i, vi)
        return vi, vj, li
    else:
        return vi, vj


def vertices_ring_faces(halfedges, sort=False, order=False, return_lengths=False):
    """Returns the face-ring (incident faces) of each vertex, as arrays
    of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.
    sort : bool
        If `True`, the central vertices' indices are sorted increasingly.
        If `False` (default), the central vertices' indices are not
        sorted but the function runs faster.
    order : bool
        If `True`, the incident faces are ordered contiguously
        counterclockwise around the central vertices.
        If `False` (default), the face-rings are not ordered contiguously
        but the function runs faster.
    return_lengths : bool (optional)
        if `True`, the function returns also the number of vertices in each
        vertex-ring (default is `False`).

    Returns
    -------
    vi : np.array
        The indices of the central vertices. Each index appears as many
        times as the number of incident faces at the vertex.
    fj : np.array
        The indices of the corresponding ring-faces.
    """
    if order:
        i = vertices_ring_ordered_halfedges(halfedges)
        vi = halfedges[i, 0]
        fj = halfedges[i, 1]
    else:
        i = np.where(halfedges[:, 1] >= 0)[0]
        vi = halfedges[i, 0]
        fj = halfedges[i, 1]
        if sort:
            i = np.argsort(vi)
            vi = vi[i]
            fj = fj[i]
    if return_lengths:
        i = np.ones(fj.shape[0], dtype='i')
        li = sum_repeated(i, vi)
        return vi, fj, li
    else:
        return vi, fj


def edges(halfedges):
    """Returns the indices of the vertices of the mesh edges

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array (E, 2)
        An array with the indices of the vertices of the i-th edge in the
        i-th row.
    """
    v = halfedges[np.argsort(halfedges[:, 5]), 0]
    v1 = v[0::2]
    v2 = v[1::2]
    return np.column_stack((v1, v2))


def vertices_ring_edges(halfedges, sort=False, order=False):
    """Returns the edge-ring (incident edges) of each vertex, as arrays
    of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.
    sort : bool
        If `True`, the central vertices' indices are sorted increasingly.
        If `False` (default), the central vertices' indices are not
        sorted but the function runs faster.
    order : bool
        If `True`, the edge-rings are ordered contiguously counterclockwise
        around the central vertices.
        If `False` (default), the edge-rings are not ordered contiguously
        but the function runs faster.

    Returns
    -------
    vi : np.array
        The indices of the central vertices. Each index appears as many
        times as the number of incident edges at the vertex (vertex valence).
    ej : np.array
        The indices of the corresponding ring-edges.
    """
    vi = halfedges[:, 0]
    ej = halfedges[:, 5]
    if order:
        i = vertices_ring_ordered_halfedges(halfedges)
        vi = vi[i]
        ej = ej[i]
    elif sort:
        i = np.argsort(vi)
        vi = vi[i]
        ej = ej[i]
    return vi, ej


def faces_edge_vertices(halfedges, sort=False, order=False):
    """Returns the vertices of the edges bounding each face, as arrays
    of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.
    sort : bool
        If `True`, the indices of the faces are sorted increasingly.
        If `False` (default), the indices of the faces are not sorted but
        the function runs faster.
    order : bool
        If `True`, the vertices of the edges are ordered contiguously
        counterclockwise around the faces.
        If `False` (default), the vertices of the edges are not ordered
        contiguously but the function runs faster.

    Returns
    -------
    fi : np.array
        The indices of the faces. Each index appears as many times as the
        number of edges around the face.
    v1 : np.array
        The indices of the tail of the edge, pointing counterclockwise
        around the face.
    v2 : np.array
        The indices of the tip of the edge, pointing counterclockwise
        around the face.
    """
    fi = halfedges[:, 1]
    v1 = halfedges[:, 0]
    v2 = halfedges[halfedges[:, 2], 0]
    if order:
        i = faces_ordered_halfedges(halfedges)
        fi = fi[i]
        v1 = v1[i]
        v2 = v2[i]
    else:
        i = np.where(halfedges[:, 1] >= 0)[0]
        fi = fi[i]
        v1 = v1[i]
        v2 = v2[i]
        if sort:
            i = np.argsort(fi)
            v1 = v1[i]
            v2 = v2[i]
    return fi, v1, v2


def faces_vertices(halfedges):
    """Returns the vertices of each face, as arrays of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    fi : np.array
        The indices of the faces. Each index appears as many times as the
        number of vertices around the face.
    vj : np.array
        The indices of the corresponding vertices, ordered contiguously
        counterclockwise around the faces.
    """
    i = faces_ordered_halfedges(halfedges)
    vi = halfedges[i, 0]
    fi = halfedges[i, 1]
    return fi, vi


def faces_valence(halfedges):
    i = faces_ordered_halfedges(halfedges)
    return sum_repeated(np.ones(len(i)), halfedges[i, 1])


def faces_halfedges(halfedges):
    """Returns the halfedges of each face, as arrays of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    fi : np.array
        The indices of the faces. Each index appears as many times as the
        number of vertices around the face.
    hj : np.array
        The indices of the corresponding halfedges, ordered contiguously
        counterclockwise around the faces.
    """
    i = faces_ordered_halfedges(halfedges)
    fi = halfedges[i, 1]
    return fi, i


def faces_size(halfedges):
    """Returns the number of vertices of each face.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array (F,)
        An array containing in i-th position the number of vertices of
        the i-th face.
    """
    fi, _ = faces_vertices(halfedges)
    L = sum_repeated(np.ones(len(fi)), fi)
    return L


def vertices_valence(halfedges):
    """Returns the number of vertices connected to each vertex.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array (F,)
        An array containing in i-th position the number of vertices of
        connected with the i-th vertex.
    """
    _, _, L = vertices_ring_vertices(halfedges, sort=True, return_lengths=True)
    return L


def faces_edges(halfedges):
    """Returns the edges bounding each face, as arrays of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    fi : np.array
        The indices of the faces. Each index appears as many times as the
        number of edges around the face.
    ej : np.array
        The indices of the corresponding edges, ordered contiguously
        counterclockwise around the faces.
    """
    i = faces_ordered_halfedges(halfedges)
    ej = halfedges[i, 5]
    fi = halfedges[i, 1]
    return fi, ej


def edges_vertices(halfedges, sort=False):
    """Returns the vertices of the edges, as arrays of indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.
    sort : bool
        If `True`, the indices of the edges are sorted increasingly.
        If `False` (default), the indices of the edges are not sorted but
        the function runs faster.

    Returns
    -------
    e : np.array
        The indices of the edges. Each index appears twice.
    vi : np.array
        The indices of the  first and second edge's vertices.
    """
    e = halfedges[:, 5]
    vi = halfedges[:, 0]
    if sort:
        i = np.argsort(halfedges[:, 5])
        e = e[i]
        vi = vi[i]
    return e, vi


def vertices_edge_map(halfedges):
    """Returns a map from vertices indices to the corresponding edge index

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    scipy.sparse.csc_matrix (V,V)
        A matrix with the index of the edge connecting vertices i-j
        in its i,j position.
    """
    V = np.max(halfedges[:, 0])
    v1 = halfedges[:, 0]
    v2 = halfedges[halfedges[:, 4], 0]
    e = halfedges[:, 5]
    edge_map = coo_matrix((e, (v1, v2)), shape=(V, V))
    edge_map = edge_map.tocsc()
    return edge_map


def edges_halfedges(halfedges):
    """Returns the indices of the two halfedges of each edge.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array (E,2)
        An array with the indices of the two halfedges of the i-th edge in
        its i-th row.
    """
    e = np.argsort(halfedges[:, 5])
    h1 = e[0::2]
    h2 = e[1::2]
    return np.column_stack((h1, h2))


def edges_faces(halfedges):
    """Returns the indices of the two faces incident to each edge.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array (E,2)
        An array with the indices of the two faces incident to the i-th edge in
        its i-th row.
    """
    f = halfedges[np.argsort(halfedges[:, 5]), 1]
    f1 = f[0::2]
    f2 = f[1::2]
    return np.column_stack((f1, f2))


def vertices_double_ring_vertices(halfedges):
    """Returns the double vertex-ring (vertices connected by maximum two edges).

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    vi : np.array
        The indices of the central vertices.
    vj : np.array
        The indices of the corresponding double ring-vertices.
    """
    v, vj = vertices_ring_vertices(halfedges, sort=True)
    V = np.max(halfedges[:, 1])
    M = coo_matrix((vj, (v, vj)), shape=(V, V))
    M = M.todense()
    ring = np.copy(M)
    while v.shape[0] > 0:
        vi, j = np.unique(v, True)
        ring[vi] += M[vj[j]]
        v = np.delete(v, j)
        vj = np.delete(vj, j)
    return ring.nonzero()


def faces_list(halfedges):
    fi, vj = faces_vertices(halfedges)
    split = np.nonzero(fi[1:] - fi[:-1])[0] + 1
    f_list = np.split(vj, split)
    return f_list


def faces(halfedges):
    f_list = faces_list(halfedges)
    return np.array(f_list)


