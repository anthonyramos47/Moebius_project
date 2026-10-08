"""
B-spline surface utilities for the Moebius / sphere-congruence pipeline.
Replaces the old QS_project/utils/bsplines_functions.py (removed from the
tree on 2026-10-07; recoverable from git history) — Polyscope removed,
imports updated to the new hanan geometry package.
"""

import json
import os

import igl
import numpy as np
import splipy as sp
from hanan.geometry.construction import normalize_vertices
from scipy.interpolate import bisplev, bisplrep
from scipy.optimize import minimize
from scipy.spatial import KDTree

# ── I/O ──────────────────────────────────────────────────────────────────────

def read_bspline_json(path: str, normalize: bool = True,
                      normalize_factor: float = 2.0) -> sp.Surface:
    """Read a B-spline surface from a Rhino-exported JSON file.

    Parameters
    ----------
    normalize : centre and scale control points so the longest bounding-box
                dimension equals normalize_factor (default True, factor 2 →
                fits in [-2, 2]³, matching the old pipeline behaviour).
    """
    with open(path) as f:
        data = json.load(f)

    order_u = data["degreeU"] + 1
    order_v = data["degreeV"] + 1

    knots_u = np.array(data["knotsU"], dtype=float)
    knots_v = np.array(data["knotsV"], dtype=float)

    # Normalise knots to [0, 1]
    knots_u = (knots_u - knots_u[0]) / (knots_u[-1] - knots_u[0])
    knots_v = (knots_v - knots_v[0]) / (knots_v[-1] - knots_v[0])

    # Add one repeated knot at each end (Rhino exports without them)
    knots_u = np.concatenate([[knots_u[0]], knots_u, [knots_u[-1]]])
    knots_v = np.concatenate([[knots_v[0]], knots_v, [knots_v[-1]]])

    ctrl = np.array(data["controlPoints"]).reshape(-1, 4)[:, :3]

    if normalize:
        shape = ctrl.shape
        ctrl = normalize_vertices(ctrl.reshape(-1, 3), normalize_factor).reshape(shape)

    basis_u = sp.BSplineBasis(order_u, knots_u)
    basis_v = sp.BSplineBasis(order_v, knots_v)
    return sp.Surface(basis_u, basis_v, ctrl)


def save_bspline_to_json(bsp: sp.Surface, path: str) -> None:
    """Save a splipy Surface in the JSON format read_bspline_json expects.

    read_bspline_json ADDS a repeated knot at each end, because Rhino exports
    without them. So the knots written here must have those ends stripped,
    otherwise a reload adds two more, the basis implies two extra control
    points per direction, and the control-point array no longer reshapes
    (ValueError: cannot reshape array of size 1200 into shape (22,22,3)).
    Saving must be the exact inverse of reading, or the round trip breaks.

    The same goes for the control-point ordering: splipy's Surface constructor
    reads a flat (N, 3) list v-major, while ``controlpoints`` is indexed
    [u, v, :]. Flattening it directly transposes the control net on reload,
    which reads back without error but gives a different surface (max |dV| was
    2.0 on Tunel). Hence the transpose below.
    """
    cp = bsp.controlpoints.transpose(1, 0, 2).reshape(-1, 3)
    cp_homogeneous = np.hstack([cp, np.ones((len(cp), 1))])
    ku = list(bsp.bases[0].knots)[1:-1]
    kv = list(bsp.bases[1].knots)[1:-1]
    data = {
        "degreeU": bsp.order(0) - 1,
        "degreeV": bsp.order(1) - 1,
        "knotsU": ku,
        "knotsV": kv,
        "controlPoints": cp_homogeneous.flatten().tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── sampling ─────────────────────────────────────────────────────────────────

def sample_bspline_surface(bsp: sp.Surface, u_pts: np.ndarray, v_pts: np.ndarray):
    """Return (V, F) quad mesh from evaluating bsp on a u×v grid."""
    S = bsp(u_pts, v_pts).reshape(-1, 3)
    nu, nv = len(u_pts), len(v_pts)
    F = np.array([
        [nv * i + j, nv * i + j + 1, nv * (i + 1) + j + 1, nv * (i + 1) + j]
        for i in range(nu - 1) for j in range(nv - 1)
    ])
    return S, F


# ── curvature & normals ───────────────────────────────────────────────────────

def bspline_curvatures(bsp: sp.Surface, u_pts: np.ndarray, v_pts: np.ndarray):
    """Return (K, H, n) Gaussian curvature, mean curvature, normals on (u,v) grid."""
    n   = bsp.normal(u_pts, v_pts)
    su  = bsp.derivative(u_pts, v_pts, d=(1, 0))
    sv  = bsp.derivative(u_pts, v_pts, d=(0, 1))
    suu = bsp.derivative(u_pts, v_pts, d=(2, 0))
    svv = bsp.derivative(u_pts, v_pts, d=(0, 2))
    suv = bsp.derivative(u_pts, v_pts, d=(1, 1))

    E = np.einsum("ijk,ijk->ij", su,  su)
    F = np.einsum("ijk,ijk->ij", su,  sv)
    G = np.einsum("ijk,ijk->ij", sv,  sv)
    L = np.einsum("ijk,ijk->ij", suu, n)
    M = np.einsum("ijk,ijk->ij", suv, n)
    N = np.einsum("ijk,ijk->ij", svv, n)

    denom = E * G - F * F
    K = (L * N - M * M) / denom
    H = (E * N + G * L - 2 * F * M) / (2 * denom)
    return K, H, n


def normal_derivatives_uv(bsp: sp.Surface, u_pts: np.ndarray, v_pts: np.ndarray):
    """Return (nu, nv) — derivatives of the unit normal field."""
    su  = bsp.derivative(u_pts, v_pts, d=(1, 0))
    sv  = bsp.derivative(u_pts, v_pts, d=(0, 1))
    suu = bsp.derivative(u_pts, v_pts, d=(2, 0))
    svv = bsp.derivative(u_pts, v_pts, d=(0, 2))
    suv = bsp.derivative(u_pts, v_pts, d=(1, 1))

    raw_n  = np.cross(su, sv, axis=2)
    n_norm = np.linalg.norm(raw_n, axis=2, keepdims=True)

    suu_sv = np.cross(suu, sv,  axis=2)
    su_suv = np.cross(su,  suv, axis=2)
    suv_sv = np.cross(suv, sv,  axis=2)
    su_svv = np.cross(su,  svv, axis=2)

    def _dn(a, b):
        cross_sum = a + b
        return (a / n_norm + b / n_norm
                - raw_n * np.einsum("ijk,ijk->ij", cross_sum, raw_n)[..., None] / n_norm**3)

    nu = _dn(suu_sv, su_suv)
    nv = _dn(suv_sv, su_svv)
    return nu, nv


# ── sphere congruence ─────────────────────────────────────────────────────────

def central_spheres(bsp: sp.Surface, u_pts: np.ndarray, v_pts: np.ndarray):
    """Compute the central-sphere radii and normals from mean curvature."""
    _, H, n = bspline_curvatures(bsp, u_pts, v_pts)
    r_H = 1.0 / H
    s = bsp(u_pts, v_pts)
    c = s + r_H[..., None] * n
    return c, r_H, H, n


def init_sphere_congruence(mid_init: int, bsp: sp.Surface,
                           u_pts: np.ndarray, v_pts: np.ndarray,
                           sample: tuple):
    """Return (r_H, n): initial sphere radii and normals.

    mid_init=0  → radii from mean curvature  (r = 1/H)
    mid_init=1  → constant radii (r = 5)
    """
    if mid_init == 0:
        _, r_H, _, n = central_spheres(bsp, u_pts, v_pts)
    else:
        n   = bsp.normal(u_pts, v_pts)
        r_H = 5.0 * np.ones((sample[0], sample[1]))
    return r_H, n


def r_uv_fitting(u_pts: np.ndarray, v_pts: np.ndarray,
                  r_H: np.ndarray, u_degree: int = 5, v_degree: int = 5):
    """Fit a scipy bivariate spline r(u,v) to the radius values r_H."""
    U, V = np.meshgrid(u_pts, v_pts, indexing="ij")
    return bisplrep(U.ravel(), V.ravel(), r_H.ravel(), kx=u_degree, ky=v_degree)


def sphere_congruence_derivatives(bsp: sp.Surface, r_uv, u_pts, v_pts):
    """Return (cu, cv) — derivatives of c(u,v) = s(u,v) + r(u,v)*n(u,v)."""
    su = bsp.derivative(u_pts, v_pts, d=(1, 0))
    sv = bsp.derivative(u_pts, v_pts, d=(0, 1))
    n  = bsp.normal(u_pts, v_pts)
    nu, nv = normal_derivatives_uv(bsp, u_pts, v_pts)

    r  = bisplev(u_pts, v_pts, r_uv)
    ru = bisplev(u_pts, v_pts, r_uv, dx=1, dy=0)
    rv = bisplev(u_pts, v_pts, r_uv, dx=0, dy=1)

    cu = su + ru[..., None] * n + r[..., None] * nu
    cv = sv + rv[..., None] * n + r[..., None] * nv
    return cu, cv


def line_congruence_uv(bsp: sp.Surface, r_uv, u_pts: np.ndarray, v_pts: np.ndarray):
    """Return (nu,nv,3) array of unit line-congruence vectors l = cu × cv / ||·||."""
    cu, cv = sphere_congruence_derivatives(bsp, r_uv, u_pts, v_pts)
    l = np.cross(cu, cv, axis=2)
    l /= np.linalg.norm(l, axis=2, keepdims=True)
    return l


def flip_lc(l: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Ensure l points in the same half-space as the surface normal n."""
    return np.sign(np.einsum("ijk,ijk->ij", l, n))[..., None] * l


# ── remeshing helpers ─────────────────────────────────────────────────────────

def _barycentric_tri(p, a, b, c):
    """igl barycentric coordinates, across libigl versions.

    libigl 2.6 renamed `barycentric_coordinates_tri` to
    `barycentric_coordinates` (the `_tet` overload moved under the same name).
    Prefer the current name and fall back for pre-2.6 installs.
    """
    fn = getattr(igl, "barycentric_coordinates", None) or \
         getattr(igl, "barycentric_coordinates_tri")
    return fn(p, a, b, c)


def interpolate_lc(foot_pts: np.ndarray, V: np.ndarray, TF: np.ndarray,
                   l: np.ndarray) -> np.ndarray:
    """Barycentric interpolation of line congruence to new foot points."""
    _, f_idx, cpts = igl.point_mesh_squared_distance(foot_pts, V, TF)
    v0, v1, v2 = V[TF[f_idx, 0]], V[TF[f_idx, 1]], V[TF[f_idx, 2]]
    bar = _barycentric_tri(cpts, v0, v1, v2)
    l0, l1, l2 = l[TF[f_idx, 0]], l[TF[f_idx, 1]], l[TF[f_idx, 2]]
    return bar[:, 0:1] * l0 + bar[:, 1:2] * l1 + bar[:, 2:3] * l2
