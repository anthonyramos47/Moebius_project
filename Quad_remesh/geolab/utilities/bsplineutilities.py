# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

def bspline_basis_functions(t, degree, knot_vector):
    """Returns the B-spline basis functions evaluated at parameters t.

    Parameters
    ----------
    t : np.array (n,)
        The array of parameters.

    degree : int
        The degree of the B-spline.

    knot_vector : np.array (m,)
        The array of the knot vector.

    Returns
    -------
    np.array (n, m - degree - 1)
        An array that contains the j-th basis function evaluated at t_i in
        position (i, j).

    """
    m = len(knot_vector) - degree - 2
    N = np.zeros((len(t), m + 2))
    for i in range(m):
        c1 = t >= knot_vector[i]
        c2 = t < knot_vector[i + 1]
        ind = np.logical_and(c1, c2)
        N[ind, i] = 1
    c1 = t >= knot_vector[m]
    c2 = t <= knot_vector[m + 1]
    ind = np.logical_and(c1, c2)
    N[ind, m] = 1
    for r in range(1, degree + 1):
        for j in range(m + 1):
            a = knot_vector[j + r] - knot_vector[j]
            if a != 0:
                A = (t - knot_vector[j]) / a
            else:
                A = np.zeros(len(t))
            b = knot_vector[j + r + 1] - knot_vector[j + 1]
            if b != 0:
                B = (knot_vector[j + r + 1] - t) / b
            else:
                B = np.zeros(len(t))
            N[:, j] = A * N[:, j] + B * N[:, j + 1]
    N = np.delete(N, -1, axis=1)
    return N


def bspline_derivative_basis_functions(t, degree, knot_vector, d=1):
    """Returns the d derivative of a B-spline evaluated at parameters t.

    Parameters
    ----------
    t : np.array (n,)
        The array of parameters.

    degree : int
        The degree of the B-spline.

    knot_vector : np.array (m,)
        The array of the knot vector.

    d : int
        The order of the derivative.

    Returns
    -------
    np.array (n, m - degree - 1 - d)
        An array that contains the j-th basis function of the derivative
        evaluated at t_i in position (i, j).

    """
    m = len(knot_vector) - degree - 2
    if degree < d:
        return np.zeros((len(t), m + 1))
    N = bspline_basis_functions(t, degree - d, knot_vector[d:-d])
    for n in range(d):
        i = np.arange(m + 1 - (d - n - 1)) + (d - n - 1)
        a = (knot_vector[i + degree - (d - n - 1)] - knot_vector[i])
        zero = np.where(a == 0)[0]
        a[zero] = 1
        f = (degree - (d - n - 1)) / a
        f[zero] = 0
        f = np.hstack((f, [0]))
        N = np.column_stack((np.zeros(len(N)), N, np.zeros(len(N))))
        N = np.einsum('j,ij->ij', f, N)
        N = N[:, 0:-1] - N[:, 1:]
    return N
