"""
Spherical-panel geometry utilities.

A spherical panel is a geodesic patch on a sphere (c, r) whose boundary is
the intersection of the sphere with a set of tangent planes (one per edge).
For visualisation we approximate the panel as a triangle fan from the sphere
centre projected onto the panel boundary curve.

The main entry point is:

    V, F = spherical_panels_from_mesh(v_mesh, f_mesh, centers, radii)

which returns a single (V, F) triangle mesh with one panel per quad face.

Optionally, the function can be used via kayviz:

    import kayviz as kv
    kv.register_surface_mesh("panels", *spherical_panels_from_mesh(...))
"""

from __future__ import annotations
import numpy as np


def _geodesic_arc(c: np.ndarray, r: float,
                  p0: np.ndarray, p1: np.ndarray,
                  n_pts: int = 8) -> np.ndarray:
    """Discretise the great-circle arc from p0 to p1 on sphere (c, r)."""
    a = (p0 - c) / r
    b = (p1 - c) / r
    # orthogonalise b with respect to a
    b -= np.dot(b, a) * a
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-10:
        return np.stack([p0, p1])
    b /= b_norm
    theta = np.arccos(np.clip(np.dot((p0 - c) / r, (p1 - c) / r), -1, 1))
    ts = np.linspace(0, theta, n_pts)
    arc = c + r * (np.cos(ts)[:, None] * a + np.sin(ts)[:, None] * b)
    return arc


def spherical_panel_from_loop(c: np.ndarray, r: float,
                               loop_pts: np.ndarray,
                               n_arc: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Build a triangulated spherical panel from a boundary loop on sphere (c, r).

    Parameters
    ----------
    c        : sphere centre (3,)
    r        : sphere radius  (scalar)
    loop_pts : (k, 3) boundary vertices already on the sphere
    n_arc    : arc discretisation per edge

    Returns
    -------
    V : (nv, 3)  vertices
    F : (nf, 3)  triangle faces
    """
    k = len(loop_pts)
    boundary = []
    for i in range(k):
        arc = _geodesic_arc(c, r, loop_pts[i], loop_pts[(i + 1) % k], n_arc)
        boundary.append(arc[:-1])          # omit last (== next start)
    boundary_pts = np.concatenate(boundary, axis=0)

    # Fan triangulation from the centroid projected onto the sphere
    raw_centre = boundary_pts.mean(axis=0)
    panel_centre = c + r * (raw_centre - c) / np.linalg.norm(raw_centre - c)

    V = np.vstack([panel_centre[None], boundary_pts])
    n_b = len(boundary_pts)
    F = np.stack([
        np.zeros(n_b, dtype=np.int32),
        np.arange(1, n_b + 1, dtype=np.int32),
        np.roll(np.arange(1, n_b + 1, dtype=np.int32), -1),
    ], axis=1)
    return V, F


def spherical_panels_from_mesh(
    v_mesh: np.ndarray,
    f_mesh: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    n_arc: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one spherical panel per quad face.

    For each face f = [i0, i1, i2, i3], the panel boundary consists of the
    four mesh vertices projected onto the corresponding sphere (centers[f], radii[f]).

    Parameters
    ----------
    v_mesh  : (nv, 3) mesh vertices
    f_mesh  : (nf, 4) quad face indices
    centers : (nf, 3) sphere centres
    radii   : (nf,)   sphere radii

    Returns
    -------
    V : (NV, 3)  all panel vertices concatenated
    F : (NF, 3)  all panel triangle faces (with global vertex offsets applied)
    """
    all_V = []
    all_F = []
    offset = 0
    for fi, face in enumerate(f_mesh):
        c = centers[fi]
        r = radii[fi]
        raw_pts = v_mesh[face]
        # Project boundary vertices onto the sphere
        vecs = raw_pts - c
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        loop_pts = c + r * vecs / (norms + 1e-12)
        Vi, Fi = spherical_panel_from_loop(c, r, loop_pts, n_arc)
        all_V.append(Vi)
        all_F.append(Fi + offset)
        offset += len(Vi)
    return np.concatenate(all_V, axis=0), np.concatenate(all_F, axis=0)


def sphere_centers_radii_from_params(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Convert algebraic sphere params (A, B, C) to (centers, radii).

    The sphere equation is  A‖v‖² - B·v + C = 0  (A ≠ 0).
    center = B / (2A),   radius = √(‖B‖²/(4A²) - C/A).
    """
    centers = B.reshape(-1, 3) / (2 * A[:, None])
    r2 = np.einsum("ij,ij->i", centers, centers) - C / A
    radii = np.sqrt(np.maximum(r2, 0))
    return centers, radii
