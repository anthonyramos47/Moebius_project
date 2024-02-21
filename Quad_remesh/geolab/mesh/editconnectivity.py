"""
Connectivity edits
"""

__author__ = 'Davide Pellis'

import numpy as np

from geolab.mesh.halfedges import are_halfedges


def join_meshes(vertices, connectivity):
    """Join several meshes.

    Parameters
    ----------
    vertices : list or tuple [n]
        A list containing the arrays of vertices, as np.array (V, 3).

    connectivity : list or tuple [n]
        A list containing the connectivity: array of faces or halfedges.

    Returns
    -------
    V : np.array (v, 3)
        The array of vertices of the joined mesh
    C : np.array
        The array of connectivity of the joined mesh
    """
    nV = [len(v) for v in vertices]
    nV[-1] = 0
    if are_halfedges(vertices[0], connectivity[0]):
        nH = [len(h) for h in connectivity]
        nH[-1] = 0
        C = []
        offset = np.zeros(6)
        for i in range(len(connectivity)):
            boundary = np.where(connectivity[i][:, 1] < 0)[0]
            C.append(connectivity[i] + offset)
            C[-1][boundary, 1] = -1
            offset[0] += len(vertices[i])
            offset[1] = np.max(C[-1][:, 1]) + 1
            offset[2:5] += len(C[-1])
            offset[5] = np.max(C[-1][:, 5]) + 1
        C = np.vstack(C)
    else:
        C = []
        for i in range(len(connectivity)):
            C.append(connectivity[i] + nV[i - 1])
        C = np.vstack(C)
    V = np.vstack(vertices)
    return V, C.astype('i')

