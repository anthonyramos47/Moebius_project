"""
Algebraic ("implicit") spheres and the initial sphere congruence for stage 5.

A sphere is carried as psi(x) = A<x,x> - <B,x> + C = 0, normalised by

    <B,B> - 4 A C = 1                      (energies.SphereUnit)

under which
    centre c = B / (2A),      radius r = 1 / (2A).

The form is projective: it represents planes as well as spheres (A = 0 gives
-<B,x> + C = 0, a plane with unit normal B), which is exactly why it is used
instead of storing (c, r) directly.
"""

import numpy as np
from scipy.interpolate import bisplev

from moebius.utils.bsplines import sample_bspline_surface

# A sphere with |A| below this is treated as a plane; r = 1/(2A) is then
# larger than 1/(2*PLANE_TOL) and the centre is numerically meaningless.
PLANE_TOL = 1e-8


def implicit_sphere_from_center_radius(c, r, atol: float = 1e-9):
    """(centres, radii) -> (A, B, C) with <B,B> - 4AC = 1.

        A = 1/(2r),   B = 2 A c = c/r,   C = (<B,B> - 1) / (4A)

    Degenerate inputs are rejected rather than silently producing inf/nan:

    r = 0   a point, not a sphere. A = 1/(2r) is infinite and the normalisation
            <B,B> - 4AC = 1 cannot hold, so there is no implicit form. Use the
            point directly.
    r = inf a plane. A = 0, and the plane needs a NORMAL, which (c, r) does not
            carry — c runs off to infinity with r. Build it with
            `implicit_plane(normal, point)` instead.
    """
    c = np.atleast_2d(np.asarray(c, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if len(c) != len(r):
        raise ValueError(f"{len(c)} centres but {len(r)} radii")

    zero = np.abs(r) <= atol
    if zero.any():
        raise ValueError(
            f"{int(zero.sum())} of {len(r)} radii are zero (|r| <= {atol:g}): a "
            "point has no implicit sphere form (A = 1/(2r) is infinite). "
            f"First at index {int(np.flatnonzero(zero)[0])}."
        )
    big = ~np.isfinite(r)
    if big.any():
        raise ValueError(
            f"{int(big.sum())} of {len(r)} radii are infinite: that is a plane, "
            "and (c, r) cannot express which plane. Use implicit_plane(normal, "
            "point)."
        )

    A = 1.0 / (2.0 * r)
    B = (2.0 * A)[:, None] * c
    C = (np.einsum("ij,ij->i", B, B) - 1.0) / (4.0 * A)

    # vectorised normalisation check (the original looped in Python)
    unit = np.einsum("ij,ij->i", B, B) - 4.0 * A * C
    bad = ~np.isclose(unit, 1.0)
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            f"<B,B> - 4AC != 1 on {int(bad.sum())} spheres; first at {k}: "
            f"{unit[k]:.6g} (r = {r[k]:.6g})"
        )
    return A, B, C


def implicit_plane(normal, point):
    """Plane as an implicit 'sphere': A = 0, B = unit normal, C = <B, point>."""
    n = np.atleast_2d(np.asarray(normal, dtype=float))
    p = np.atleast_2d(np.asarray(point, dtype=float))
    nn = np.linalg.norm(n, axis=1)
    if np.any(nn <= 0):
        raise ValueError("zero-length plane normal")
    B = n / nn[:, None]                     # <B,B> - 4*0*C = 1 automatically
    A = np.zeros(len(B))
    C = np.einsum("ij,ij->i", B, p)
    return A, B, C


def center_radius_from_implicit(A, B, C):
    """(A, B, C) -> (centres, radii, is_plane).

    r = sqrt(<B,B> - 4AC) / (2A). The square root is kept rather than assuming
    the normalised value 1 because SphereUnit is a SOFT constraint: mid-
    optimisation <B,B> - 4AC only sits NEAR 1, and using the real value keeps
    the radius consistent with the actual coefficients. It is clamped at 0
    because that soft constraint can also dip slightly negative, which would
    otherwise produce a silent nan.

    Faces with |A| <= PLANE_TOL are planes: centre is nan, radius inf.
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float)).reshape(len(A), 3)
    C = np.atleast_1d(np.asarray(C, dtype=float))

    is_plane = np.abs(A) <= PLANE_TOL
    safe_A = np.where(is_plane, 1.0, A)

    centres = B / (2.0 * safe_A)[:, None]
    disc = np.einsum("ij,ij->i", B, B) - 4.0 * A * C
    radii = np.sqrt(np.maximum(disc, 0.0)) / (2.0 * safe_A)

    centres[is_plane] = np.nan
    radii[is_plane] = np.inf
    return centres, radii, is_plane
