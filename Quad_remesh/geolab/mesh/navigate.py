"""
Mesh navigation functions
"""

__author__ = 'Davide Pellis'

import numpy as np

# -------------------------------------------------------------------------
#                                Navigating
# -------------------------------------------------------------------------


def _process_sequence(sequence):
    index = {'o': 0, 'f': 1, 'n': 2, 'p': 3, 't': 4, 'e': 5}
    step = sequence.split('-')
    indices = []
    for key in step:
        try:
            indices.append(index[key])
        except KeyError:
            raise KeyError("The sequence must be a string of 'o', 'f',"
                  " 'p', 'n', and 'e' separated by '-'.")
    return indices


def navigate_halfedges(halfedges, sequence):
    """Navigates the halfedges with a string sequence.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges
    sequence : str
        A keyword string representing the navigation sequence separated by '-'
        with: o = origin, f = face, p = previous, n = next, t = twin, e = edge.
        Example: 'n-t-o' asks for the origin of the twin of the next halfedge.

    Returns
    -------
    np.array (H,)
        The indices of the last key of the sequence.
    """
    indices = _process_sequence(sequence)
    h_copy = np.copy(halfedges)
    for i in indices[:-1]:
        if i < 2 or i == 5:
            return h_copy[:, i]
        h_copy = halfedges[h_copy[:, i]]
    return h_copy[:, indices[-1]]


def halfedges_origin(halfedges):
    """Returns the index of halfedges origin.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges

    Returns
    -------
    np.array (H,)
        The indices of the origin vertices of the corresponding halfedges.
    """
    return halfedges[:, 0]


def halfedges_face(halfedges):
    """Returns the index of halfedges face.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges

    Returns
    -------
    np.array (H,)
        The indices of the faces of the corresponding halfedges.
    """
    return halfedges[:, 1]


def halfedges_next(halfedges):
    """Returns the index of halfedges next halfedge.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges

    Returns
    -------
    np.array (H,)
        The indices of next halfedges
    """
    return halfedges[:, 2]


def halfedges_previous(halfedges):
    """Returns the index of halfedges' previous halfedge.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The indices of halfedges

    Returns
    -------
    np.array (H,)
        The indices of the corresponding previous halfedges.
    """
    return halfedges[:, 3]


def halfedges_twin(halfedges):
    """Returns the index of halfedges' twin halfedge.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges

    Returns
    -------
    np.array (H,)
        The indices of the corresponding twin halfedges.
    """
    return halfedges[:, 4]


def halfedges_edge(halfedges):
    """Returns the index of halfedges edge.

    Parameters
    ----------
    halfedges : np.array (H, 6)
        The array of halfedges

    Returns
    -------
    np.array (H,)
        The indices of the corresponding edge.
    """
    return halfedges[:, 5]
