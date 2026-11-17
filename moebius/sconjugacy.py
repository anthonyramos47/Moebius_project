"""
S-conjugacy pipeline — spheres on the faces of a quad mesh whose edges follow
S-conjugate directions of a sphere congruence.

Same idea as the L-conjugacy pipeline of Laguerre_project
(Lconjugacy/pipeline.py), with spheres instead of oriented planes, and the
post-optimisation of the Moebius pipeline without the support structure.

Stage overview
--------------
0. load_sconj_surface        read B-spline JSON, set the u/v range
1. compute_radius            r = lam / H at the samples, fitted by a spline r(u, v)
2. compute_directions        principal frame (e1, e2, k1, k2) and, from a first
                             direction t1 at angle theta to e1, its S-conjugate
                             t2:  (1 - r k1) a1 b1 + (1 - r k2) a2 b2 = 0
3. export_sconjugate_field   triangulated grid + D1/D2 for quadRemesher
   remesh                    run quadRemesher and load Q
   (pipeline.fair_remeshed_surface: glide + fairness, Q -> Q')
4. project_mesh_onto_bspline vertices of Q onto the B-spline
   init_sphere_mesh          face barycentres -> (u, v) on the B-spline ->
                             sphere s_f with centre B + r n, radius r
5. setup_sconj_postopt       E_sphere, E_unit, E_fair, E_reg, E_prox_f, E_prox_C
   run_sconj_postopt         (no support structure: no E_supp, no E_torsal_plane)
6. save_sconj_results        OBJ files and state pickle

Conventions
-----------
The normal is n = unit(Su x Sv), as splipy's Surface.normal. The principal
curvatures are the eigenvalues of the shape operator I^-1 II with
II = <S_xy, n>, so H = (k1 + k2) / 2 has the sign of moebius.utils.bsplines,
and the sphere with centre S + r n is tangent to the surface with curvature
1/r. Hence r = 1/H is the mean-curvature sphere and 1 - r k_i = 0 exactly
when the sphere is a curvature sphere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import RectBivariateSpline, bisplev
from scipy.spatial import cKDTree

from hanan.geometry.io import read_obj, write_obj, triangulate_quads
from hanan.geometry.mesh import Mesh
from hanan.optimization import Optimizer

from moebius.energies import SphereFit, SphereUnit, ProximityCenters
from moebius.pipeline import (
    MoebiusState, sample_selection, run_remesher, remesh_quality,
    frame_field_outliers, _project_into_plane, _write_vec_file, _save_state,
    OUT_DIR, BSPLINE_DIR,
)
from moebius.utils.bsplines import (
    read_bspline_json, sample_bspline_surface, bspline_curvatures,
)
from moebius.utils.spheres import (
    implicit_sphere_from_center_radius, center_radius_from_implicit,
)

SCONJ_OUT_DIR = OUT_DIR / "sconjugacy"


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class SConjugacyState(MoebiusState):
    """MoebiusState plus the S-conjugacy data.

    Subclassing keeps every field the shared pipeline functions read
    (run_remesher, fair_remeshed_surface, _save_state), so they work unchanged.
    The Moebius-only fields (lc_opt, torsal_opt, l_*) simply stay None.
    """
    # Radius function r = lam * r_H, r_H = 1/H (stage 1)
    lam:      float = 1.0
    H:        Optional[np.ndarray] = None     # (nu, nv) mean curvature at the samples
    r_fit_err: float = 0.0                    # max |r_uv - lam/H| at the samples

    # Directions (stage 2)
    theta_method: str = "constant"            # "constant" | "linearU" | "linearV"
    theta_deg:    float = 15.0                # constant angle of t1 to e1
    theta_range:  tuple = (0.0, 60.0)         # start/end for linearU / linearV
    e1: Optional[np.ndarray] = None           # (N, 3) principal frame at the samples
    e2: Optional[np.ndarray] = None
    k1: Optional[np.ndarray] = None           # (N,) principal curvatures, k1 <= k2
    k2: Optional[np.ndarray] = None
    t1: Optional[np.ndarray] = None           # (N, 3) first direction
    t2: Optional[np.ndarray] = None           # (N, 3) its S-conjugate
    theta: Optional[np.ndarray] = None        # (N,) angle of t1 to e1, radians
    phi:   Optional[np.ndarray] = None        # (N,) angle of t2 to e1, radians

    # Sphere mesh (stage 4)
    vertex_uv: Optional[np.ndarray] = None    # (nv, 2) parameters of Q's vertices
    face_uv:   Optional[np.ndarray] = None    # (nf, 2) parameters of the barycentres
    c_face:    Optional[np.ndarray] = None    # (nf, 3) initial sphere centres
    r_face:    Optional[np.ndarray] = None    # (nf,)   initial signed radii
    sphere_dual_faces: Optional[list] = None  # dual combinatorics: face rings


# ── Stage 0: load ─────────────────────────────────────────────────────────────

def load_sconj_surface(name: str, experiment: str = "experiment0",
                       bspline_dir=BSPLINE_DIR, out_dir=SCONJ_OUT_DIR,
                       **settings) -> SConjugacyState:
    """Read <bspline_dir>/<name>.json and sample it (as pipeline.load_surface)."""
    state = SConjugacyState(name=name, experiment_name=experiment,
                            bspline_dir=str(bspline_dir), output_dir=str(out_dir))
    for k, v in settings.items():
        if not hasattr(state, k):
            raise AttributeError(f"Unknown SConjugacyState field: '{k}'")
        setattr(state, k, v)
    state.bspline = read_bspline_json(os.path.join(state.bspline_dir, name + ".json"))
    state.save_path = os.path.join(state.output_dir, name, experiment)
    sample_selection(state, state.interval_u, state.interval_v)
    return state


def flip_normals(state: SConjugacyState) -> None:
    """Reverse the u direction of the control net, which flips the normals."""
    state.bspline.controlpoints = state.bspline.controlpoints[::-1, :, :].copy()
    state.normalsFlipped = not state.normalsFlipped


def swap_uv(state: SConjugacyState) -> None:
    """Exchange u and v (also flips the normals), with the ranges and samples."""
    bsp = state.bspline
    bsp.swap()
    state.interval_u, state.interval_v = state.interval_v, state.interval_u
    state.u_sample_num, state.v_sample_num = state.v_sample_num, state.u_sample_num
    sample_selection(state)
    state.uvSwapped = not state.uvSwapped


# ── B-spline evaluation at scattered (u, v) ──────────────────────────────────
#
# splipy's Surface.__call__ evaluates on a TENSOR grid, so a list of k
# parameter pairs would cost a k x k grid. These evaluate pointwise from the
# basis matrices instead.

def _eval_points(bsp, u, v, du: int = 0, dv: int = 0) -> np.ndarray:
    """S^(du, dv)(u_k, v_k) for each k; (k, 3)."""
    Bu = np.asarray(bsp.bases[0].evaluate(np.asarray(u, float), d=du))
    Bv = np.asarray(bsp.bases[1].evaluate(np.asarray(v, float), d=dv))
    return np.einsum("ki,kj,ijd->kd", Bu, Bv, bsp.controlpoints)


def normal_at(bsp, u, v) -> np.ndarray:
    """Unit normal unit(Su x Sv) at scattered (u_k, v_k); (k, 3)."""
    n = np.cross(_eval_points(bsp, u, v, 1, 0), _eval_points(bsp, u, v, 0, 1))
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def project_onto_bspline(bsp, P, interval_u=(0.0, 1.0), interval_v=(0.0, 1.0),
                         seed_res: int = 150, max_iter: int = 30,
                         tol: float = 1e-13):
    """Closest points of `P` on the B-spline.

    Seeds each point with the nearest sample of a seed_res x seed_res grid over
    the (u, v) range (padded by 5% and clipped to [0, 1], since the remeshed
    boundary can sit slightly outside the sampled range), then refines with
    Gauss-Newton on |S(u, v) - p|^2, clamped to [0, 1]^2.

    Returns (uv (k, 2), S(uv) (k, 3), distance (k,)).
    """
    if bsp.rational:
        raise NotImplementedError("rational B-splines are not supported")
    P = np.asarray(P, float)

    def padded(iv):
        pad = 0.05 * (iv[1] - iv[0])
        return np.linspace(max(0.0, iv[0] - pad), min(1.0, iv[1] + pad), seed_res)

    us, vs = padded(interval_u), padded(interval_v)
    grid = bsp(us, vs).reshape(-1, 3)
    _, idx = cKDTree(grid).query(P)
    u = us[idx // len(vs)].copy()
    v = vs[idx % len(vs)].copy()

    for _ in range(max_iter):
        S  = _eval_points(bsp, u, v)
        Su = _eval_points(bsp, u, v, 1, 0)
        Sv = _eval_points(bsp, u, v, 0, 1)
        d  = P - S
        a, b, c = (np.einsum("ij,ij->i", Su, Su), np.einsum("ij,ij->i", Su, Sv),
                   np.einsum("ij,ij->i", Sv, Sv))
        g1, g2 = np.einsum("ij,ij->i", Su, d), np.einsum("ij,ij->i", Sv, d)
        det = a * c - b * b
        du = ( c * g1 - b * g2) / det
        dv = (-b * g1 + a * g2) / det
        u_new = np.clip(u + du, 0.0, 1.0)
        v_new = np.clip(v + dv, 0.0, 1.0)
        step = np.maximum(np.abs(u_new - u), np.abs(v_new - v)).max()
        u, v = u_new, v_new
        if step < tol:
            break

    S = _eval_points(bsp, u, v)
    return np.column_stack([u, v]), S, np.linalg.norm(P - S, axis=1)


# ── Stage 1: radius function ─────────────────────────────────────────────────

def compute_radius(state: SConjugacyState, lam: Optional[float] = None,
                   degree: int = 3, smoothing: float = 0.0) -> None:
    """r = lam / H at the samples (lam = 1: the mean-curvature sphere), fitted
    by a bivariate spline r(u, v) stored in state.r_uv as a (tx, ty, c, kx, ky)
    tck, the form moebius.pipeline and bisplev read.

    The fit is RectBivariateSpline on the sampling grid; smoothing = 0
    interpolates. state.r_H holds the sampled values lam / H.
    """
    if lam is not None:
        state.lam = float(lam)
    K, H, n = bspline_curvatures(state.bspline, state.u_pts, state.v_pts)
    state.H = H
    state.n_surf = n

    if np.abs(H).min() < 1e-8:
        raise ValueError(
            f"H vanishes at a sample (min |H| = {np.abs(H).min():.2e}), so r = lam/H "
            "is undefined: choose another range, or run the H != 0 B-spline "
            "optimisation first.")
    if (H > 0).any() and (H < 0).any():
        print(f"  WARNING: H changes sign on the range (min {H.min():.3f}, "
              f"max {H.max():.3f}); r = lam/H passes through infinity there.")

    r = state.lam / H
    state.r_H = r
    spl = RectBivariateSpline(state.u_pts, state.v_pts, r,
                              kx=degree, ky=degree, s=smoothing)
    tx, ty, c = spl.tck
    state.r_uv = [tx, ty, c, degree, degree]
    state.r_fit_err = float(np.abs(bisplev(state.u_pts, state.v_pts, state.r_uv) - r).max())


def radius_at(state: SConjugacyState, u, v) -> np.ndarray:
    """Fitted r(u_k, v_k) at scattered parameters; (k,)."""
    u = np.atleast_1d(np.asarray(u, float))
    v = np.atleast_1d(np.asarray(v, float))
    # bisplev evaluates on a grid; one call per point keeps it scattered.
    return np.array([bisplev(a, b, state.r_uv) for a, b in zip(u, v)])


def radius_grid(state: SConjugacyState) -> np.ndarray:
    """Fitted r on the sampling grid, flattened u-major; (N,)."""
    return np.asarray(bisplev(state.u_pts, state.v_pts, state.r_uv)).ravel()


def sphere_center_surface(state: SConjugacyState):
    """Centres S + r n of the fitted congruence on the sampling grid, and faces."""
    V, F = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    n = state.bspline.normal(state.u_pts, state.v_pts).reshape(-1, 3)
    return V + radius_grid(state)[:, None] * n, F


# ── Stage 2: principal frame and S-conjugate directions ─────────────────────

def principal_frame_points(bsp, u, v):
    """Principal frame at scattered (u_k, v_k).

    Eigen-decomposition of the shape operator I^-1 II. Returns e1, e2 (k, 3)
    unit principal directions with k1 <= k2 and (e1, e2, n) positively
    oriented, the curvatures k1, k2 (k,), and n (k, 3).
    """
    Su  = _eval_points(bsp, u, v, 1, 0)
    Sv  = _eval_points(bsp, u, v, 0, 1)
    Suu = _eval_points(bsp, u, v, 2, 0)
    Suv = _eval_points(bsp, u, v, 1, 1)
    Svv = _eval_points(bsp, u, v, 0, 2)
    n = np.cross(Su, Sv)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    dot = lambda a, b: np.einsum("ij,ij->i", a, b)
    I  = np.stack([np.stack([dot(Su, Su), dot(Su, Sv)], -1),
                   np.stack([dot(Su, Sv), dot(Sv, Sv)], -1)], -2)
    II = np.stack([np.stack([dot(Suu, n), dot(Suv, n)], -1),
                   np.stack([dot(Suv, n), dot(Svv, n)], -1)], -2)

    # I^-1 II is self-adjoint w.r.t. I, so with I = L L^T the matrix
    # L^-1 II L^-T is symmetric: eigh, real and sorted ascending.
    L = np.linalg.cholesky(I)
    Li = np.linalg.inv(L)
    Wsym = Li @ II @ np.swapaxes(Li, -1, -2)
    kappa, Y = np.linalg.eigh(Wsym)
    X = np.swapaxes(Li, -1, -2) @ Y          # eigenvectors of I^-1 II, in (Su, Sv)

    e1 = X[:, 0, 0][:, None] * Su + X[:, 1, 0][:, None] * Sv
    e2 = X[:, 0, 1][:, None] * Su + X[:, 1, 1][:, None] * Sv
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    flip = dot(np.cross(e1, e2), n) < 0
    e2[flip] *= -1
    return e1, e2, kappa[:, 0], kappa[:, 1], n


def sconjugate(e1, e2, k1, k2, r, theta):
    """S-conjugate partner of t1 = cos(theta) e1 + sin(theta) e2.

    With a = (cos theta, sin theta) and M = diag(1 - r k1, 1 - r k2) the
    condition a^T M b = 0 is solved by b = J M a, J the rotation by 90 deg:

        b = (-(1 - r k2) sin theta, (1 - r k1) cos theta)

    Returns t1, t2 (k, 3), phi (angle of t2 to e1) and |M a|, which is zero
    only where the sphere is a curvature sphere in both principal directions
    (an umbilic with r = 1/k): there t2 is undefined.

    t2 is parallel to t1 when a^T M a = 0, i.e. t1 is self-conjugate. For the
    mean-curvature sphere (r = 1/H) M is a multiple of diag(1, -1), so the
    partner of theta is 90 deg - theta and theta = 45 deg is degenerate
    everywhere. For lam = 0 (r = 0) M = I and t2 is orthogonal to t1.
    """
    a1, a2 = np.cos(theta), np.sin(theta)
    m1, m2 = 1.0 - r * k1, 1.0 - r * k2
    b1, b2 = -m2 * a2, m1 * a1
    norm = np.hypot(b1, b2)
    safe = np.where(norm > 1e-14, norm, 1.0)
    b1, b2 = b1 / safe, b2 / safe
    t1 = a1[:, None] * e1 + a2[:, None] * e2
    t2 = b1[:, None] * e1 + b2[:, None] * e2
    return t1, t2, np.arctan2(b2, b1), norm


def sconjugacy_residual(k1, k2, r, theta, phi) -> np.ndarray:
    """(1 - r k1) a1 b1 + (1 - r k2) a2 b2 for unit a, b at angles theta, phi."""
    return ((1 - r * k1) * np.cos(theta) * np.cos(phi)
            + (1 - r * k2) * np.sin(theta) * np.sin(phi))


def _theta_at(state: SConjugacyState, u, v) -> np.ndarray:
    """First-direction angle (radians) at scattered (u, v), per theta_method."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    if state.theta_method == "constant":
        return np.full(len(u), np.deg2rad(state.theta_deg))
    t0, t1 = np.deg2rad(state.theta_range[0]), np.deg2rad(state.theta_range[1])
    if state.theta_method == "linearU":
        s = (u - state.interval_u[0]) / (state.interval_u[1] - state.interval_u[0])
    elif state.theta_method == "linearV":
        s = (v - state.interval_v[0]) / (state.interval_v[1] - state.interval_v[0])
    else:
        raise ValueError(f"unknown theta_method '{state.theta_method}' "
                         "(constant, linearU, linearV)")
    return t0 + (t1 - t0) * np.clip(s, 0.0, 1.0)


def directions_at(state: SConjugacyState, u, v) -> dict:
    """Principal frame, t1 and its S-conjugate t2 at scattered (u, v)."""
    e1, e2, k1, k2, n = principal_frame_points(state.bspline, u, v)
    r = radius_at(state, u, v)
    theta = _theta_at(state, u, v)
    t1, t2, phi, norm = sconjugate(e1, e2, k1, k2, r, theta)
    return dict(e1=e1, e2=e2, k1=k1, k2=k2, n=n, r=r, theta=theta, phi=phi,
                t1=t1, t2=t2, norm=norm)


def _grid_uv(state: SConjugacyState):
    U, V = np.meshgrid(state.u_pts, state.v_pts, indexing="ij")
    return U.ravel(), V.ravel()


def _frame_angle_deg(t1, t2) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", t1, t2)), 0, 1)))


def compute_directions(state: SConjugacyState, method: Optional[str] = None,
                       theta_deg: Optional[float] = None,
                       theta_range: Optional[tuple] = None,
                       verbose: bool = True) -> None:
    """Principal frame at the sample points, t1 at angle theta to e1, and its
    S-conjugate t2 with respect to the fitted radius function."""
    if state.r_uv is None:
        raise RuntimeError("Fit the radius function first (compute_radius).")
    if method is not None:      state.theta_method = method
    if theta_deg is not None:   state.theta_deg = float(theta_deg)
    if theta_range is not None: state.theta_range = tuple(theta_range)

    d = directions_at(state, *_grid_uv(state))
    for k in ("e1", "e2", "k1", "k2", "t1", "t2", "theta", "phi"):
        setattr(state, k, d[k])

    if verbose:
        res = sconjugacy_residual(d["k1"], d["k2"], d["r"], d["theta"], d["phi"])
        ang = _frame_angle_deg(d["t1"], d["t2"])
        print(f"S-conjugacy residual max {np.abs(res).max():.2e}")
        print(f"angle(t1, t2)  min {ang.min():.1f} deg  mean {ang.mean():.1f} deg  "
              f"max {ang.max():.1f} deg")
        if ang.min() < 5:
            print(f"  WARNING: {int((ang < 5).sum())} samples have near-parallel "
                  "t1, t2 - t1 is close to self-conjugate there.")
        if (d["norm"] < 1e-10).any():
            print(f"  WARNING: {int((d['norm'] < 1e-10).sum())} samples where the "
                  "sphere is a curvature sphere in both directions: t2 undefined.")


def cross_segments(P, D, scale):
    """Curve network of segments [p - s d, p + s d] (for the viewer)."""
    n = len(P)
    verts = np.empty((2 * n, 3))
    verts[0::2] = P - scale * D
    verts[1::2] = P + scale * D
    return verts, np.stack([np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)], axis=1)


# ── Stage 3: frame field for the remesher ─────────────────────────────────────

def sconjugate_field_per_quad(state: SConjugacyState):
    """t1, t2 at the parameter midpoint of each quad of the sampling grid.

    Evaluated there rather than averaged from the corners, so no sign or
    ordering alignment of the four corner frames is needed. Quads are ordered
    as sample_bspline_surface's faces (u-major).
    """
    um = 0.5 * (state.u_pts[:-1] + state.u_pts[1:])
    vm = 0.5 * (state.v_pts[:-1] + state.v_pts[1:])
    U, V = np.meshgrid(um, vm, indexing="ij")
    d = directions_at(state, U.ravel(), V.ravel())
    return d["t1"], d["t2"]


def export_sconjugate_field(state: SConjugacyState, basename: str = "remesh_input",
                            one_frame_per_quad: bool = True,
                            filter_outliers: bool = True,
                            **outlier_kwargs) -> dict:
    """Write the triangulated sampling grid and the S-conjugate frame field.

    Same files and conventions as pipeline.export_frame_field (D1.dat, D2.dat,
    indices.dat, '--format vec', '--defined_on faces'): each quad gives two
    triangles, and with one_frame_per_quad only the one needing the smaller
    in-plane rotation is constrained. filter_outliers drops degenerate frames
    and frames disagreeing with their neighbours (pipeline.frame_field_outliers;
    the torsal-specific criteria do not apply here).
    """
    os.makedirs(state.save_path, exist_ok=True)
    V, F_quad = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    F_tri = np.array(triangulate_quads(F_quad.tolist()), dtype=np.int32)
    t1, t2 = sconjugate_field_per_quad(state)
    n_quads = len(F_quad)

    t1_tri, t2_tri = np.repeat(t1, 2, axis=0), np.repeat(t2, 2, axis=0)
    nrm = np.cross(V[F_tri[:, 1]] - V[F_tri[:, 0]], V[F_tri[:, 2]] - V[F_tri[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    p1, rot1 = _project_into_plane(t1_tri, nrm)
    p2, rot2 = _project_into_plane(t2_tri, nrm)
    rot = np.maximum(rot1, rot2)

    if one_frame_per_quad:
        pick = np.argmin(rot.reshape(n_quads, 2), axis=1)
        tri_idx = 2 * np.arange(n_quads, dtype=np.int32) + pick.astype(np.int32)
        keep_tri_of_quad = lambda keep: keep
    else:
        tri_idx = np.arange(len(F_tri), dtype=np.int32)
        keep_tri_of_quad = lambda keep: np.repeat(keep, 2)
    p1, p2, rot = p1[tri_idx], p2[tri_idx], rot[tri_idx]

    info = None
    if filter_outliers:
        kw = dict(max_neighbour_deg=35.0, min_frame_angle_deg=5.0)
        kw.update(outlier_kwargs)
        info = frame_field_outliers(state, t1, t2, drop_no_real_torsal=False,
                                    max_torsal_residual=None, **kw)
        keep = keep_tri_of_quad(info["mask"])
        tri_idx, p1, p2, rot = tri_idx[keep], p1[keep], p2[keep], rot[keep]

    paths = {"obj": os.path.join(state.save_path, basename + ".obj"),
             "d1": os.path.join(state.save_path, "D1.dat"),
             "d2": os.path.join(state.save_path, "D2.dat"),
             "indices": os.path.join(state.save_path, "indices.dat")}
    write_obj(paths["obj"], V, F_tri)
    _write_vec_file(paths["d1"], p1, "D1 - first direction t1")
    _write_vec_file(paths["d2"], p2, "D2 - S-conjugate direction t2")
    with open(paths["indices"], "w") as f:
        f.write(" ".join(str(int(i)) for i in tri_idx) + "\n")
    state.frame_field_paths = paths

    ang = _frame_angle_deg(t1, t2)
    print(f"Exported {len(V)} vertices, {len(F_tri)} triangles -> {state.save_path}")
    print(f"  {len(tri_idx)} frames constrained ({n_quads} quads)")
    if info is not None:
        for why, cnt in info["reasons"].items():
            if cnt:
                print(f"     dropped {cnt:5d}  {why}")
    print(f"  angle(t1,t2)  min {ang.min():.1f}  mean {ang.mean():.1f}  max {ang.max():.1f} deg")
    if len(rot):
        print(f"  in-plane rotation  mean {rot.mean():.1f}  max {rot.max():.1f} deg")
    else:
        print("  WARNING: no frame left to constrain the remesher - the field is "
              "degenerate everywhere (at lam = 1, theta = 45 deg gives t2 = t1).")
    return {**paths, "n_quads": n_quads, "n_frames": len(tri_idx)}


def load_remeshed(state: SConjugacyState, path: Optional[str] = None) -> None:
    """Load the remeshed polygon OBJ into state.V_remesh / F_remesh(_quads)."""
    if path is None:
        path = os.path.join(state.save_path, state.name + "_Remeshed.obj")
    V, F_list = read_obj(path)
    state.V_remesh, state.F_remesh = np.asarray(V, float), F_list
    sizes, counts = np.unique([len(f) for f in F_list], return_counts=True)
    state.F_remesh_quads = (np.array(F_list, dtype=np.int32)
                            if len(sizes) == 1 and sizes[0] == 4 else None)
    print(f"Loaded {path}\n  {len(V)} vertices, {len(F_list)} faces  "
          + ", ".join(f"{c}x{s}-gon" for s, c in zip(sizes, counts)))
    if state.F_remesh_quads is None:
        print("  NOTE: not all faces are quads; fairing and post-optimisation need "
              "an all-quad mesh. Adjust --gradient / --constraints.")


def remesh(state: SConjugacyState, gradient: float = 40.0, constraints: float = 1.0,
           extractor: str = "qex", batch: bool = True, **kwargs) -> dict:
    """pipeline.run_remesher, then load Q (without the Moebius line congruence)."""
    res = run_remesher(state, gradient=gradient, constraints=constraints,
                       extractor=extractor, batch=batch, load=False, **kwargs)
    if res["returncode"] == 0 and res["out_obj"] and os.path.isfile(res["out_obj"]):
        load_remeshed(state, res["out_obj"])
        remesh_quality(state)
    return res


def sweep_gradient(state: SConjugacyState, gradients, **kwargs) -> list:
    """Remesh at each gradient and report face counts; the state keeps the last."""
    rows = []
    for g in gradients:
        print(f"\n-- gradient {g} ------------------------------")
        res = remesh(state, gradient=g, verbose=False, **kwargs)
        rows.append(remesh_quality(state) if res["returncode"] == 0
                    else {"gradient": g, "n_faces": None})
    return rows


# ── Stage 4: sphere mesh ─────────────────────────────────────────────────────

def project_mesh_onto_bspline(state: SConjugacyState) -> np.ndarray:
    """Move every vertex of Q to its closest point on the B-spline.

    Stores the parameters in state.vertex_uv and returns the distances moved.
    """
    uv, S, dist = project_onto_bspline(state.bspline, state.V_remesh,
                                       state.interval_u, state.interval_v)
    state.V_remesh = S
    state.vertex_uv = uv
    print(f"Projected {len(S)} vertices onto the B-spline: moved median "
          f"{np.median(dist):.2e}, max {dist.max():.2e}")
    return dist


def init_sphere_mesh(state: SConjugacyState) -> None:
    """One sphere per face of Q, from the fitted congruence at its barycentre.

    The barycentre of face f is projected onto the B-spline, giving (u_f, v_f);
    the sphere s_f has centre B(u_f, v_f) + r(u_f, v_f) n(u_f, v_f) and signed
    radius r(u_f, v_f). The spheres live on the faces of Q, so the sphere mesh
    has the dual combinatorics: its vertices are Q's faces and its faces are
    the rings of faces around Q's inner vertices (state.sphere_dual_faces).
    """
    F = state.F_remesh_quads
    if F is None:
        raise RuntimeError("The sphere mesh needs an all-quad remeshed mesh.")
    bary = state.V_remesh[F].mean(axis=1)
    uv, S, _ = project_onto_bspline(state.bspline, bary,
                                    state.interval_u, state.interval_v)
    n = normal_at(state.bspline, uv[:, 0], uv[:, 1])
    r = radius_at(state, uv[:, 0], uv[:, 1])
    if (np.abs(r) < 1e-9).any():
        raise ValueError("Zero radius at some faces (lam = 0?): a point has no "
                         "sphere form. Choose lam > 0.")
    state.face_uv = uv
    state.c_face = S + r[:, None] * n
    state.r_face = r

    mesh = Mesh()
    mesh.make_mesh(state.V_remesh, F)
    state.sphere_dual_faces = [list(map(int, f)) for f in mesh.dual_top()]
    print(f"{len(F)} spheres; dual mesh with {len(state.sphere_dual_faces)} faces; "
          f"|r| median {np.median(np.abs(r)):.4f}  range "
          f"[{np.abs(r).min():.4f}, {np.abs(r).max():.4f}]")


def sphere_fit_error(state: SConjugacyState, V=None, A=None, B=None, C=None):
    """max_k | |v_k - c_f| - |r_f| | per face (geometric form of E_sphere)."""
    if A is None:
        if state.postopt is not None:
            A, B, C = state.postopt.unpack("A", "B", "C")
            V = state.postopt.unpack("v").reshape(-1, 3)
        else:
            A, B, C = implicit_sphere_from_center_radius(state.c_face, state.r_face)
    V = state.V_remesh if V is None else V
    c, r, _ = center_radius_from_implicit(A, np.asarray(B).reshape(-1, 3), C)
    F = state.F_remesh_quads
    return np.abs(np.linalg.norm(V[F] - c[:, None], axis=2) - np.abs(r)[:, None]).max(axis=1)


# ── Stage 5: post-optimisation ────────────────────────────────────────────────

def setup_sconj_postopt(state: SConjugacyState,
                        w_sphere: float = 1.0,
                        w_unit: float = 10.0,
                        w_fair: float = 1e-1,
                        w_prox: float = 1e-4,
                        w_glide: float = 1e-4,
                        w_prox_c: float = 1e-5,
                        prox_epsilon: float = 0.1,
                        w_reg_v: float = 0.0,
                        w_reg_s: float = 0.0,
                        warmup_iters: int = 10,
                        fix_boundary: bool = True) -> Optimizer:
    """The Moebius post-optimisation without the support structure.

    E_sphere  SphereFit: each sphere passes through its face's vertices
    E_unit    SphereUnit: <B,B> - 4AC = 1, keeps (A, B, C) off zero
    E_fair    Chakana's quad-mesh fairness on the vertices
    E_reg     step dampening (control_variable) on the vertices (w_reg_v) and
              on the sphere coefficients (w_reg_s); 0 by default, for the
              reason given in pipeline.setup_postopt_optimizer
    E_prox_f  vertices -> reference surface, closest point and tangent plane
    E_prox_C  sphere centres -> the centre surface S + r n of the fitted
              congruence

    No E_supp and no E_torsal_plane: the sphere centres need not form planar
    rings, so there are no nd, l, n_l variables.
    """
    if state.c_face is None:
        raise RuntimeError("Initialise the sphere mesh first (init_sphere_mesh).")
    V = state.V_remesh
    F = state.F_remesh_quads
    A0, B0, C0 = implicit_sphere_from_center_radius(state.c_face, state.r_face)

    if warmup_iters > 0:
        wu = Optimizer()
        wu.add_variable("A", A0)
        wu.add_variable("B", B0.ravel())
        wu.add_variable("C", C0)
        wu.add_objective_term(SphereFit(),  (F, V), w=w_sphere, ce=True)
        wu.add_objective_term(SphereUnit(), (),     w=w_unit,   ce=True)
        wu.initialize_optimizer(verbose=False, adaptive_mu=False)
        wu.optimize(max_iter=warmup_iters)
        A0, B0, C0 = wu.unpack("A", "B", "C")
        print(f"Warm-up: {wu.it} sphere-only steps, "
              f"energy {wu.energy[0]:.4e} -> {wu.energy[-1]:.4e}")

    opt = Optimizer()
    opt.add_variable("v", V.ravel().copy())
    opt.add_variable("A", np.asarray(A0, float))
    opt.add_variable("B", np.asarray(B0, float).ravel())
    opt.add_variable("C", np.asarray(C0, float))

    opt.add_objective_term(SphereFit(),  (F,), w=w_sphere, ce=True)
    opt.add_objective_term(SphereUnit(), (),   w=w_unit,   ce=True)

    V_ref, F_ref_quad = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    F_ref = np.array(triangulate_quads(F_ref_quad.tolist()), dtype=np.int32)
    opt.proximity_reference("v", V_ref, F_ref, w_prox, w_glide)
    C_ref, _ = sphere_center_surface(state)
    opt.add_objective_term(ProximityCenters(), (C_ref, F_ref, prox_epsilon),
                           w=w_prox_c, ce=True)

    mesh = Mesh()
    mesh.make_mesh(V, F)
    opt.set_fairness("v", mesh.vertex_adjacency_list(), dim=3, w=w_fair)
    if w_reg_v:
        opt.control_variable("v", w_reg_v)
    if w_reg_s:
        for name in ("A", "B", "C"):
            opt.control_variable(name, w_reg_s)
    if fix_boundary:
        bnd = np.asarray(mesh.boundary_vertices()).ravel()
        opt.fix_variables("v", bnd, dim=3)
        print(f"Holding {len(bnd)} boundary vertices fixed")

    opt.initialize_optimizer(verbose=True, adaptive_mu=False)
    state.postopt = opt
    return opt


def run_sconj_postopt(state: SConjugacyState, max_iter: int = 50) -> None:
    state.postopt.optimize(max_iter=state.postopt.it + max_iter)
    print(state.postopt.get_report())


def sphere_mesh_centers(state: SConjugacyState):
    """(centres, dual faces) of the current sphere mesh, for viewing / export."""
    if state.postopt is not None:
        A, B, C = state.postopt.unpack("A", "B", "C")
        c, _, _ = center_radius_from_implicit(A, B.reshape(-1, 3), C)
    else:
        c = state.c_face
    return c, state.sphere_dual_faces


# ── Stage 6: save ─────────────────────────────────────────────────────────────

def save_sconj_results(state: SConjugacyState) -> None:
    """Write OBJ files and pickle the state into state.save_path."""
    from moebius.glyphs import panels_from_spheres
    os.makedirs(state.save_path, exist_ok=True)
    out = lambda f: os.path.join(state.save_path, f)

    V, F = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    write_obj(out("surface.obj"), V, F)
    Vc, _ = sphere_center_surface(state)
    write_obj(out("sphere_centers.obj"), Vc, F)
    if state.V_remesh is not None:
        write_obj(out("remeshed.obj"), state.V_remesh, state.F_remesh)
    if state.sphere_dual_faces is not None:
        c, dual = sphere_mesh_centers(state)
        write_obj(out("sphere_mesh_centers.obj"), c, dual)
    if state.postopt is not None:
        V_opt = state.postopt.unpack("v").reshape(-1, 3)
        write_obj(out("optimised_mesh.obj"), V_opt, state.F_remesh)
        A, B, C = state.postopt.unpack("A", "B", "C")
        pV, pF = panels_from_spheres(V_opt, state.F_remesh_quads, A, B.reshape(-1, 3), C)
        write_obj(out("sphere_panels.obj"), pV, pF)
    _save_state(state, "state.pickle")
    print(f"Results saved to {state.save_path}")
