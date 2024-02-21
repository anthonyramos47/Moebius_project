"""
A mesh obj reader
"""

__author__ = 'Davide Pellis'

import numpy as np

# -----------------------------------------------------------------------------

from geolab.utilities.stringutilities import make_filepath

from geolab.mesh.globalconnectivity import faces_list

# -----------------------------------------------------------------------------


def number_of_faces(halfedges):
    """Computes the number of faces.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    int
        The number of faces as the highest face index.
    """
    return np.max(halfedges[:, 1]) + 1


def number_of_vertices(halfedges):
    """Computes the number of vertices.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    int
        The number of vertices as the highest vertex index.
    """
    return np.max(halfedges[:, 0]) + 1


def number_of_edges(halfedges):
    """Computes the number of edges.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges.

    Returns
    -------
    int
        The number of edges as the highest edge index.
    """
    return np.max(halfedges[:, 5]) + 1





