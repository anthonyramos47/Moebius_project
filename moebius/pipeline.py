"""
Moebius sphere-congruence pipeline — notebook-friendly, GUI-free.

Stage overview
--------------
0. load_surface          read B-spline JSON, set u/v range, compute reference
1. compute_init          initial sphere congruence r(u,v) and line congruence l
2. setup_lc_optimizer    build Optimizer for LC + LC-orth step
   run_lc_optimizer      run it
3. setup_torsal_optimizer add torsal terms to the optimiser
   run_torsal_optimizer  run it
4. save_remesh_input     export cross-field OBJ for the remesher
   load_remeshed_surface load remeshed quad mesh
5. setup_postopt_optimizer sphere + support + regularity energies
   run_postopt_optimizer run it
6. save_results          dump OBJ files and state pickle
"""

from __future__ import annotations

import os
import pickle
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import igl
import splipy as sp

from hanan.geometry.mesh import Mesh
from hanan.geometry.io import write_obj, read_obj, triangulate_quads
from hanan.optimization import Optimizer

from moebius.utils.bsplines import (
    read_bspline_json,
    sample_bspline_surface,
    normal_derivatives_uv,
    init_sphere_congruence,
    r_uv_fitting,
    line_congruence_uv,
    flip_lc,
    interpolate_lc,
    bspline_curvatures,
)
from moebius.utils.mesh_utils import lc_info_at_grid_points, torsal_directions, unit

from moebius.energies import (
    MeanCurvatureBspline,
    LineCong, LineCongOrth, Torsal, TorsalAngle,
    SphereFit, SupportPlanarity, RegFaces,
)

HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE.parent / "data"
BSPLINE_DIR = DATA_DIR / "bsplines"
OUT_DIR     = HERE.parent / "notebooks" / "out"

DEFAULT_REMESHER = str(Path.home() / "KAUST" / "QuadRemesher" /
                       "Quad_Remesher" / "build" / "quadRemesher")

# Default optimisation weights -------------------------------------------------
LC_WEIGHTS = {"LineCong": 1.0, "LineCongOrth": 1.0}
TORSAL_WEIGHTS = {
    "LineCong": 1.0, "LineCongOrth": 0.5,
    "Torsal": 1.0, "TorsalAngle": 0.5,
}
POSTOPT_WEIGHTS = {
    "SphereFit": 1.0, "SupportPlanarity": 1e-1,
    "RegFaces": 1e-2,
}


# ── State dataclass ────────────────────────────────────────────────────────────

@dataclass
class MoebiusState:
    # Surface
    name: str = ""
    bspline: object = None
    normalsFlipped: bool = False
    uvSwapped:      bool = False

    # Sampling
    u_sample_num: int = 20
    v_sample_num: int = 20
    interval_u: list = field(default_factory=lambda: [0.0, 1.0])
    interval_v: list = field(default_factory=lambda: [0.0, 1.0])
    u_pts: Optional[np.ndarray] = None
    v_pts: Optional[np.ndarray] = None

    # Sphere congruence initialisation
    init_mode:   int   = 0        # 0 = from mean curvature, 1 = constant r=5
    r_uv:        object = None    # scipy bivariate spline (t, c, k)
    r_H:         Optional[np.ndarray] = None
    n_surf:      Optional[np.ndarray] = None
    l_init:      Optional[np.ndarray] = None

    # Optimisation angle constraints
    angle_lc:     float = 85.0   # degrees — how close l can be to n
    angle_torsal: float = 30.0   # degrees — minimum torsal-plane angle

    # Optimisers (carry the variable state)
    lc_opt:      Optional[Optimizer] = None
    torsal_opt:  Optional[Optimizer] = None
    postopt:     Optional[Optimizer] = None

    # Remeshed data
    V_remesh: Optional[np.ndarray] = None
    F_remesh: Optional[np.ndarray] = None
    l_remesh: Optional[np.ndarray] = None

    # Paths
    experiment_name: str = "experiment0"
    bspline_dir:  str = str(BSPLINE_DIR)
    output_dir:   str = str(OUT_DIR)
    save_path:    str = ""
    remesher_bin: str = DEFAULT_REMESHER


# ── Stage 0: load ─────────────────────────────────────────────────────────────

def load_surface(name: str, experiment: str = "experiment0",
                 bspline_dir=BSPLINE_DIR, out_dir=OUT_DIR,
                 **settings) -> MoebiusState:
    """Read <bspline_dir>/<name>.json, sample the surface."""
    state = MoebiusState(name=name, experiment_name=experiment,
                          bspline_dir=str(bspline_dir), output_dir=str(out_dir))
    for k, v in settings.items():
        if not hasattr(state, k):
            raise AttributeError(f"Unknown MoebiusState field: '{k}'")
        setattr(state, k, v)
    bsp_path = os.path.join(state.bspline_dir, name + ".json")
    state.bspline = read_bspline_json(bsp_path)
    state.save_path = os.path.join(state.output_dir, name, experiment)
    sample_selection(state, state.interval_u, state.interval_v)
    return state


def sample_selection(state: MoebiusState,
                     interval_u=None, interval_v=None,
                     u_sample_num=None, v_sample_num=None) -> None:
    """Update u/v sample points from intervals."""
    if interval_u   is not None: state.interval_u   = interval_u
    if interval_v   is not None: state.interval_v   = interval_v
    if u_sample_num is not None: state.u_sample_num = u_sample_num
    if v_sample_num is not None: state.v_sample_num = v_sample_num
    state.u_pts = np.linspace(state.interval_u[0], state.interval_u[1], state.u_sample_num)
    state.v_pts = np.linspace(state.interval_v[0], state.interval_v[1], state.v_sample_num)


# ── Stage 1: initialisation ───────────────────────────────────────────────────

def compute_init(state: MoebiusState) -> None:
    """Compute initial r(u,v), l(u,v) from the B-spline surface."""
    bsp = state.bspline
    u, v = state.u_pts, state.v_pts

    r_H, n = init_sphere_congruence(state.init_mode, bsp, u, v,
                                     (state.u_sample_num, state.v_sample_num))
    state.r_H    = r_H
    state.n_surf = n
    state.r_uv   = r_uv_fitting(u, v, r_H)

    l = line_congruence_uv(bsp, state.r_uv, u, v)
    l = flip_lc(l, n)
    state.l_init = l

    print(f"Init  r range  [{r_H.min():.4f}, {r_H.max():.4f}]")
    print(f"Init  l·n mean  {np.einsum('ijk,ijk->ij',l,n).mean():.4f}")


# ── Stage 1b: B-spline mean-curvature optimisation ───────────────────────────

def setup_bspline_optimizer(state: MoebiusState,
                             H_thresh: float = 0.05,
                             w_H: float = 1.0,
                             w_step: float = 0.1) -> Optimizer:
    """Optimise the surface B-spline control points to ensure H ≠ 0.

    Builds a grid adjacency over the control points and uses:
      - MeanCurvatureBspline  — push |H| above H_thresh
      - step_control          — limit CP displacement per iteration
      - set_lap_smooth        — keep the deformation smooth
    """
    bsp = state.bspline
    u, v = state.u_pts, state.v_pts
    cp0  = bsp.controlpoints.copy().ravel()
    nu_cp, nv_cp = bsp.controlpoints.shape[:2]

    opt = Optimizer()
    opt.add_variable("cp_surf", cp0)

    H_term = MeanCurvatureBspline()
    opt.add_objective_term(H_term, (bsp, u, v, H_thresh), w=w_H, ce=True)

    opt.control_variable("cp_surf", w_step)
    opt.set_lap_smooth("cp_surf", np.arange(nu_cp * nv_cp),
                       _cp_adjacency(nu_cp, nv_cp), dim=3, w=1e-2)

    opt.initialize_optimizer(verbose=True)
    state.bspline_opt = opt
    return opt


def run_bspline_optimizer(state: MoebiusState, max_iter: int = 30) -> None:
    """Run the B-spline optimiser and apply the result to state.bspline."""
    state.bspline_opt.optimize(max_iter=max_iter)
    cp_opt = state.bspline_opt.unpack("cp_surf")
    state.bspline.controlpoints = cp_opt.reshape(state.bspline.controlpoints.shape)
    print(state.bspline_opt.get_report())


def _cp_adjacency(nu: int, nv: int) -> list:
    """4-connected grid adjacency list for a nu×nv control-point grid."""
    adj = []
    for i in range(nu):
        for j in range(nv):
            nb = []
            if i > 0:      nb.append((i - 1) * nv + j)
            if i < nu - 1: nb.append((i + 1) * nv + j)
            if j > 0:      nb.append(i * nv + j - 1)
            if j < nv - 1: nb.append(i * nv + j + 1)
            adj.append(nb)
    return adj


# ── Stage 2: LC optimisation ──────────────────────────────────────────────────

def setup_lc_optimizer(state: MoebiusState,
                       w_lc: float = 1.0,
                       w_orth: float = 1.0) -> Optimizer:
    """Build an Optimizer for the line-congruence + orthogonality step."""
    bsp = state.bspline
    u, v = state.u_pts, state.v_pts
    r_uv  = state.r_uv
    cp    = r_uv[2].copy()
    l     = state.l_init.copy()
    N     = state.u_sample_num * state.v_sample_num
    sample = (state.u_sample_num, state.v_sample_num)

    opt = Optimizer()
    opt.add_variable("rij", cp)
    opt.add_variable("l",   l.ravel())
    opt.add_variable("mu",  np.full(N, 50.0))

    lc_term = LineCong()
    opt.add_objective_term(lc_term, (bsp, r_uv, u, v), w=w_lc, ce=True)

    orth_term = LineCongOrth()
    opt.add_objective_term(orth_term, (bsp, r_uv, u, v, state.angle_lc),
                            w=w_orth, ce=True)

    opt.unitize_variable("l", 3, w=10.0)
    opt.set_fairness("l", _vertex_adj(state), dim=3, w=0.01)

    opt.initialize_optimizer(verbose=True)
    state.lc_opt = opt
    return opt


def run_lc_optimizer(state: MoebiusState, max_iter: int = 50) -> None:
    state.lc_opt.optimize(max_iter=max_iter)
    # copy r_uv control points back into state
    state.r_uv[2] = state.lc_opt.unpack("rij")
    print(state.lc_opt.get_report())


# ── Stage 3: torsal optimisation ──────────────────────────────────────────────

def setup_torsal_optimizer(state: MoebiusState,
                            w_lc: float = 1.0,
                            w_orth: float = 0.5,
                            w_torsal: float = 1.0,
                            w_tangle: float = 0.5) -> Optimizer:
    """Build an Optimizer that includes torsal-direction terms."""
    bsp = state.bspline
    u, v = state.u_pts, state.v_pts
    r_uv  = state.r_uv
    sample = (state.u_sample_num, state.v_sample_num)
    n_sq  = (state.u_sample_num - 1) * (state.v_sample_num - 1)

    # Carry over previous LC solution
    prev = state.lc_opt
    f_l, f_cp, f_mu = prev.unpack("l", "rij", "mu")
    n_flat = state.n_surf.reshape(-1, 3)
    f_l = np.sign(np.einsum("ij,ij->i", f_l.reshape(-1, 3), n_flat))[:, None] * f_l.reshape(-1, 3)

    opt = Optimizer()
    opt.add_variable("rij",  f_cp)
    opt.add_variable("l",    f_l.ravel())
    opt.add_variable("mu",   f_mu)
    opt.add_variable("nt1",  np.zeros(3 * n_sq))
    opt.add_variable("nt2",  np.zeros(3 * n_sq))
    opt.add_variable("u1",   np.zeros(n_sq))
    opt.add_variable("v1",   np.zeros(n_sq))
    opt.add_variable("u2",   np.zeros(n_sq))
    opt.add_variable("v2",   np.zeros(n_sq))
    opt.add_variable("theta", np.zeros(n_sq))

    lc_term = LineCong()
    opt.add_objective_term(lc_term, (bsp, r_uv, u, v), w=w_lc, ce=True)

    orth_term = LineCongOrth()
    opt.add_objective_term(orth_term, (bsp, r_uv, u, v, state.angle_lc),
                            w=w_orth, ce=True)

    torsal_term = Torsal()
    opt.add_objective_term(torsal_term,
                            (bsp, u, v, state.n_surf, sample),
                            w=w_torsal, ce=True)

    tangle_term = TorsalAngle()
    opt.add_objective_term(tangle_term, (state.angle_torsal,),
                            w=w_tangle, ce=True)

    opt.unitize_variable("l",   3, w=10.0)
    opt.unitize_variable("nt1", 3, w=10.0)
    opt.unitize_variable("nt2", 3, w=10.0)
    opt.control_variable("nt1", 0.05)
    opt.control_variable("nt2", 0.05)
    opt.set_fairness("l", _vertex_adj(state), dim=3, w=0.01)

    opt.initialize_optimizer(verbose=True)
    state.torsal_opt = opt
    return opt


def run_torsal_optimizer(state: MoebiusState, max_iter: int = 100) -> None:
    state.torsal_opt.optimize(max_iter=max_iter)
    state.r_uv[2] = state.torsal_opt.unpack("rij")
    print(state.torsal_opt.get_report())


# ── Stage 4: remeshing ────────────────────────────────────────────────────────

def save_remesh_input(state: MoebiusState) -> str:
    """Write OBJ + cross-field file for the external quad remesher."""
    os.makedirs(state.save_path, exist_ok=True)
    V, F = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)

    obj_path = os.path.join(state.save_path, "remesh_input.obj")
    cf_path  = os.path.join(state.save_path, "crossfield.txt")

    write_obj(obj_path, V, F)

    # Export per-face torsal direction as cross-field
    opt   = state.torsal_opt
    l, u1, v1, u2, v2 = opt.unpack("l", "u1", "v1", "u2", "v2")
    l = l.reshape(state.u_sample_num, state.v_sample_num, 3)
    lc, lu, lv = lc_info_at_grid_points(l)
    lu = lu.reshape(-1, 3); lv = lv.reshape(-1, 3)
    t1 = u1[:, None] * lu + v1[:, None] * lv
    t2 = u2[:, None] * lu + v2[:, None] * lv
    # Normalise for the remesher
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12

    with open(cf_path, "w") as f:
        for i in range(len(t1)):
            f.write(f"{t1[i,0]} {t1[i,1]} {t1[i,2]} "
                    f"{t2[i,0]} {t2[i,1]} {t2[i,2]}\n")

    # Pickle full state for later
    _save_state(state, "pre_remesh_state.pickle")
    print(f"Remesh input saved to {obj_path}")
    return obj_path


def run_remesher(state: MoebiusState, n_faces: int = 400) -> int:
    """Call the external quad-remesher binary."""
    obj_in  = os.path.join(state.save_path, "remesh_input.obj")
    cf_in   = os.path.join(state.save_path, "crossfield.txt")
    obj_out = os.path.join(state.save_path, "remeshed.obj")
    cmd = [state.remesher_bin, obj_in, cf_in, obj_out, str(n_faces)]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def load_remeshed_surface(state: MoebiusState) -> None:
    """Load remeshed OBJ and interpolate line-congruence to its vertices."""
    obj_path = os.path.join(state.save_path, "remeshed.obj")
    V, F_list = read_obj(obj_path)
    state.V_remesh = V
    state.F_remesh = np.array(F_list, dtype=np.int32)

    # Build triangle mesh from sampling grid for IGL-based interpolation
    V_orig, F_quad = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    F_tri = np.array(triangulate_quads(F_quad.tolist()), dtype=np.int32)

    l_flat = state.torsal_opt.unpack("l").reshape(-1, 3)
    state.l_remesh = interpolate_lc(V, V_orig, F_tri, l_flat)
    print(f"Loaded remeshed mesh: {len(V)} vertices, {len(state.F_remesh)} faces")


# ── Stage 5: post-optimisation ────────────────────────────────────────────────

def setup_postopt_optimizer(state: MoebiusState,
                             w_sphere: float = 1.0,
                             w_support: float = 1e-1,
                             w_reg: float = 1e-2) -> Optimizer:
    """Build the post-optimisation Optimizer (sphere fit + support + regularity)."""
    V = state.V_remesh
    F = state.F_remesh

    mesh = Mesh()
    mesh.make_mesh(V, F)
    vertex_sph  = F                         # vertex indices per sphere face (nf, 4)
    sph_sph_adj = mesh.dual_top()           # ordered face rings around each inner vertex
    inner_v     = mesh.inner_vertices()

    # Interior edge → face and vertex pairs
    ie       = mesh.inner_edges()           # shape (E_inner,)
    ev1, ev2 = mesh.edge_vertices()         # (E_all,), (E_all,)
    ef1, ef2 = mesh.edge_faces()            # (E_all,), (E_all,)  (-1 = boundary)
    e_f_f    = (ef1[ie], ef2[ie])
    e_v_v    = (ev1[ie], ev2[ie])

    # Sphere params from current geometry
    n_f  = len(F)
    A0   = np.ones(n_f)
    B0   = np.zeros((n_f, 3))
    for fi, f in enumerate(F):
        pts = V[f]
        B0[fi] = pts.mean(axis=0)
    C0 = np.zeros(n_f)

    # Declare ALL variables first (Jacobian shape is fixed at len(X) on first add_objective_term)
    opt = Optimizer()
    opt.add_variable("v",  V.ravel())
    opt.add_variable("A",  A0)
    opt.add_variable("B",  B0.ravel())
    opt.add_variable("C",  C0)
    opt.add_variable("nd", np.zeros(3 * len(inner_v)))

    sph_term  = SphereFit()
    supp_term = SupportPlanarity()
    reg_term  = RegFaces()
    opt.add_objective_term(sph_term,  (vertex_sph,),      w=w_sphere,  ce=True)
    opt.add_objective_term(supp_term, (sph_sph_adj,),     w=w_support, ce=True)
    opt.add_objective_term(reg_term,  (e_f_f, e_v_v),     w=w_reg,     ce=True)

    opt.set_fairness("v", mesh.vertex_adjacency_list(), dim=3, w=1e-3)
    opt.initialize_optimizer(verbose=True)
    state.postopt = opt
    return opt


def run_postopt_optimizer(state: MoebiusState, max_iter: int = 50) -> None:
    state.postopt.optimize(max_iter=max_iter)
    print(state.postopt.get_report())


# ── Stage 6: save ─────────────────────────────────────────────────────────────

def save_results(state: MoebiusState) -> None:
    """Write OBJ files and pickle the full state."""
    os.makedirs(state.save_path, exist_ok=True)

    # Surface mesh
    V, F = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    write_obj(os.path.join(state.save_path, "surface.obj"), V, F)

    # Sphere-centre surface (offset by r(u,v) along normal)
    from scipy.interpolate import bisplev
    r  = bisplev(state.u_pts, state.v_pts, state.r_uv)
    n  = state.bspline.normal(state.u_pts, state.v_pts).reshape(-1, 3)
    Vc = V + r.ravel()[:, None] * n
    write_obj(os.path.join(state.save_path, "sphere_centers.obj"), Vc, F)

    # Remeshed surface (if available)
    if state.V_remesh is not None:
        write_obj(os.path.join(state.save_path, "remeshed.obj"),
                   state.V_remesh, state.F_remesh)

    # Post-opt mesh (if available)
    if state.postopt is not None:
        V_opt = state.postopt.unpack("v").reshape(-1, 3)
        write_obj(os.path.join(state.save_path, "optimised_mesh.obj"),
                   V_opt, state.F_remesh)

        from moebius.glyphs import sphere_centers_radii_from_params, spherical_panels_from_mesh
        A, B, C = state.postopt.unpack("A", "B", "C")
        B = B.reshape(-1, 3)
        centers, radii = sphere_centers_radii_from_params(A, B, C)
        pV, pF = spherical_panels_from_mesh(V_opt, state.F_remesh, centers, radii)
        write_obj(os.path.join(state.save_path, "sphere_panels.obj"), pV, pF)

    _save_state(state, "state.pickle")
    print(f"Results saved to {state.save_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_state(state: MoebiusState, filename: str) -> None:
    os.makedirs(state.save_path, exist_ok=True)
    path = os.path.join(state.save_path, filename)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path: str) -> MoebiusState:
    with open(path, "rb") as f:
        return pickle.load(f)


def _vertex_adj(state: MoebiusState):
    """Vertex-adjacency list for the sampling grid (used by set_fairness)."""
    V, F = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    mesh = Mesh()
    mesh.make_mesh(V, F)
    return mesh.vertex_adjacency_list()


def _tri_quads(F_quad: np.ndarray) -> np.ndarray:
    """Triangulate a quad mesh by splitting each face into two triangles."""
    return np.vstack([F_quad[:, [0, 1, 2]], F_quad[:, [0, 2, 3]]])
