"""
Mesh boundary functions
"""

__author__ = 'Davide Pellis'

import numpy as np

from geolab.mesh.globalconnectivity import faces_list

from geolab.mesh.navigate import navigate_halfedges


# -----------------------------------------------------------------------------
#                                 Boundary
# -----------------------------------------------------------------------------


def boundary_halfedges(halfedges):
    """Returns the indices of the boundary halfedges.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array
        An array with the indices of the boundary halfedges.
    """
    b = np.where(halfedges[:, 1] < 0)[0]
    return b


def inner_halfedges(halfedges):
    """Returns the indices of the interior halfedges.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array
        An array with the indices of the interior halfedges.
    """
    h = np.where(halfedges[:, 1] >= 0)[0]
    return h


def boundary_vertices(halfedges):
    """Returns the indices of the boundary vertices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array
        An array with the indices of the boundary vertices.
    """
    b = np.where(halfedges[:, 1] < 0)[0]
    v = halfedges[b, 0]
    return v


def are_boundary_vertices(halfedges, vertices=None):
    if vertices is None:
        vertices = np.arange(np.max(halfedges[:, 0] + 1))
    are_boundary = np.in1d(vertices, boundary_vertices(halfedges))
    return are_boundary


def inner_vertices(halfedges):
    """Returns the indices of the inner vertices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array
        An array with the indices of the inner vertices.
    """
    b = np.where(halfedges[:, 1] >= 0)[0]
    v = halfedges[b, 0]
    return v


def boundary_faces(halfedges):
    """Returns the indices of the faces with at least one edge on the boundary.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array
        An array with the indices of the boundary faces.
    """
    b = boundary_halfedges(halfedges)
    e = halfedges[halfedges[b, 4], 1]
    return e


def boundary_edges(halfedges):
    """Returns the indices of the boundary edges.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    np.array
        An array with the indices of the boundary faces.
    """
    e = halfedges[np.where(halfedges[:, 1] < 0)[0], 5]
    e = np.unique(e)
    return e


def boundary_contiguous_vertices(halfedges, corners=None):
    """Returns the indices of the boundary vertices ordered contiguously.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.
    corners : np.array of int
        Array containing the indices of corner vertices. Default is None.
        If specified, the curves are split at corners.

    Returns
    -------
    list of lists
        A list containing lists of indices of boundary vertices ordered
        contiguously clockwise, one for each boundary polyline.
    """
    boundaries = []
    boundary_h = []
    for h in range(halfedges.shape[0]):
        if halfedges[h, 1] < 0 and h not in boundary_h:
            boundary = []
            h_he = h
            boundary_h.append(h_he)
            boundary.append(halfedges[h_he, 0])
            h_he = halfedges[h_he, 2]
            while h_he != h:
                boundary_h.append(h_he)
                boundary.append(halfedges[h_he, 0])
                h_he = halfedges[h_he, 2]
            boundaries.append(np.array(boundary))
    if corners is not None:
        split_curves = []
        for boundary in boundaries:
            indices = np.arange(len(boundary))
            c = indices[np.in1d(boundary, corners)]
            boundary = np.split(boundary, c)
            for i in range(len(boundary) - 1):
                a = boundary[i]
                boundary[i] = np.insert(a, a.shape, boundary[i + 1][0])
            if len(boundary) > 1:
                boundary[0] = np.hstack((boundary[-1], boundary[0]))
                del boundary[-1]
            split_curves.extend(boundary)
            boundaries = split_curves
    return boundaries


def label_contiguous_boundaries(halfedges):
    """Labels different boundaries with different negative indices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    H : np.array (H, 6)
        The new array of halfedges.
    """
    H = np.copy(halfedges)
    H[H[:, 1] < 0, 1] = -99999
    bn = -1
    for h in range(H.shape[0]):
        if H[h, 1] == -99999:
            H[h, 1] = bn
            h_next = H[h, 2]
            while h_next != h:
                H[h_next, 1] = bn
                h_next = H[h_next, 2]
            bn -= 1
    return H


def fill_boundaries(halfedges, return_face_shift=False):
    """Fills the boundaries of the mesh with a single polygonal face.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    return_face_shift: bool
        It 'True', it returns the shift of old faces. Default is 'False'.

    Returns
    -------
    H : np.array (H, 6)
        The new array of halfedges.

    shift (optional) : int
        The shift of old faces.
    """
    H = label_contiguous_boundaries(halfedges)
    if return_face_shift:
        shift = np.min(H[:, 1])
        H[:, 1] -= np.min(H[:, 1])
        return H, shift
    H[:, 1] -= np.min(H[:, 1])
    return H


def extract_boundary_faces(halfedges):
    H = label_contiguous_boundaries(halfedges)
    n = - np.min(H[:, 1])
    H[:, 1] += n
    F = faces_list(H)
    return F[:n]


def boundary_contiguous_halfedges(halfedges, corners=None):
    """Returns the indices of the boundary halfedges ordered contiguously.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.
    corners : np.array of int
        Array containing the indices of corner vertices. Default is None.
        If specified, the curves are split at corners.

    Returns
    -------
    list of lists
        A list containing lists of indices of boundary halfedges ordered
        contiguously clockwise, one for each boundary polyline.
    """
    boundaries = []
    visited = []
    for h in range(halfedges.shape[0]):
        if halfedges[h, 1] < 0 and h not in visited:
            boundary = []
            h_he = h
            boundary.append(h_he)
            h_he = halfedges[h_he, 2]
            while h_he != h:
                boundary.append(h_he)
                h_he = halfedges[h_he, 2]
            boundaries.append(np.array(boundary))
            visited.extend(boundary)
    if corners is not None:
        corner_boundaries = []
        for boundary in boundaries:
            indices = np.arange(len(boundary))
            c = indices[np.in1d(halfedges[boundary, 0], corners)]
            boundary = np.split(boundary, c)
            if len(boundary) > 1:
                boundary[0] = np.hstack((boundary[-1], boundary[0]))
                del boundary[-1]
            corner_boundaries.extend(boundary)
        boundaries = corner_boundaries
    return boundaries


def mesh_corners(vertices, halfedges, corner_tolerance=0.5):
    """Returns the indices of the corner vertices.

    Parameters
    ----------
    vertices : np.array (V, 3)
        The array of vertices.
    halfedges : np.array (H, 6)
        The array of halfedges.
    corner_tolerance : float
        A value between 0 and 1 that specifies the tolerance of corners
        detection where 0 corresponds is a 0° angle between edges, 0.5 to 90°,
        and 1 to 180°.

    Returns
    -------
    np.array
        Array of indices of the corner vertices.
    """
    if corner_tolerance is None:
        return []
    tol = corner_tolerance * 2 - 1
    b = np.where(halfedges[:, 1] < 0)[0]
    v0 = halfedges[b, 0]
    vp = halfedges[halfedges[b, 3], 0]
    vn = halfedges[halfedges[b, 2], 0]
    Vp = vertices[v0, :] - vertices[vp, :]
    Vn = vertices[vn, :] - vertices[v0, :]
    Vp = Vp / (np.linalg.norm(Vp, axis=1, keepdims=True) + 1e-10)
    Vn = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True ) + 1e-10)
    C = np.einsum('ij,ij->i', Vp, Vn)
    corners = v0[np.where(C[:] <= tol)[0]]
    return corners


def double_boundary_vertices(halfedges):
    h = boundary_halfedges(halfedges)
    bv = navigate_halfedges(halfedges, 'o')[h]
    dbv = navigate_halfedges(halfedges, 't-p-o')[h]
    bound = np.hstack((bv, dbv))
    return np.unique(bound)
