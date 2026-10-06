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
4. export_frame_field    write triangulated OBJ + per-face D1/D2 files
   run_remesher          call quadRemesher (batch or viewer), load the result
   sweep_gradient        try several --gradient values to pick a density
5. setup_postopt_optimizer sphere + support + regularity energies
   run_postopt_optimizer run it
6. save_results          dump OBJ files and state pickle
"""

from __future__ import annotations

import os
import pickle
import shutil
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

PROJECT_ROOT = HERE.parent

# Resolved lazily by resolve_remesher_bin(); see REMESHER_CANDIDATES below.
DEFAULT_REMESHER = ""

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
    angle_torsal: float = 75.0   # degrees — MINIMUM torsal-plane angle.
                                 # Larger = stronger (90 = exactly orthogonal
                                 # torsal planes); useful range 45..90, 45 the
                                 # minimum. Paper used 45 and 60.

    # Optimisers (carry the variable state)
    lc_opt:      Optional[Optimizer] = None
    torsal_opt:  Optional[Optimizer] = None
    postopt:     Optional[Optimizer] = None

    # Remeshed data
    V_remesh: Optional[np.ndarray] = None
    F_remesh: Optional[list] = None            # ragged: list of index lists
    F_remesh_quads: Optional[np.ndarray] = None  # (nf,4) iff every face is a quad
    l_remesh: Optional[np.ndarray] = None

    # Inequality formulation: True = hinge residuals (no mu/theta slacks)
    hinge: bool = True

    # Remeshing inputs / settings
    frame_field_paths: Optional[dict] = None   # {'obj','d1','d2'}
    remesh_gradient:    float = 40.0           # higher = denser quads
    remesh_constraints: float = 1.0            # fraction of constrained faces

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

    opt.initialize_optimizer(verbose=True, adaptive_mu=False)
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
                       w_orth: float = 2.0,
                       hinge: bool = True) -> Optimizer:
    """Build an Optimizer for the line-congruence + orthogonality step.

    `hinge=True` (default) states the angle constraint as the one-sided
    residual max(0, cos^2 a - <l,n>^2) and drops the `mu` slack entirely.
    `hinge=False` restores the published dummy-variable form.
    """
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
    if not hinge:
        # Smooth form needs mu consistent with the residual
        # (l.n)^2 - cos^2 a - mu^2; the old constant 50 made every residual
        # ~ -2500 and the initial energy ~1.4e9, pure artefact.
        n_flat  = state.n_surf.reshape(-1, 3)
        ln2     = np.einsum("ij,ij->i", l.reshape(-1, 3), n_flat) ** 2
        cos2a   = np.cos(np.deg2rad(state.angle_lc)) ** 2
        opt.add_variable("mu", np.sqrt(np.maximum(ln2 - cos2a, 0.0)))

    lc_term = LineCong()
    opt.add_objective_term(lc_term, (bsp, r_uv, u, v), w=w_lc, ce=True)

    orth_term = LineCongOrth()
    opt.add_objective_term(orth_term, (bsp, r_uv, u, v, state.angle_lc, hinge),
                            w=w_orth, ce=True)

    opt.unitize_variable("l", 3, w=10.0)
    opt.set_fairness("l", _vertex_adj(state), dim=3, w=0.01)

    opt.initialize_optimizer(verbose=True, adaptive_mu=False)
    state.lc_opt = opt
    state.hinge = hinge
    return opt


def run_lc_optimizer(state: MoebiusState, max_iter: int = 50) -> None:
    state.lc_opt.optimize(max_iter=max_iter)
    # copy r_uv control points back into state
    state.r_uv[2] = state.lc_opt.unpack("rij")
    print(state.lc_opt.get_report())


# ── Stage 3: torsal optimisation ──────────────────────────────────────────────

def setup_torsal_optimizer(state: MoebiusState,
                            w_lc: float = 1.0,
                            w_orth: float = 0.01,
                            w_torsal: float = 1.0,
                            w_tangle: float = 2.0,
                            hinge: Optional[bool] = None) -> Optimizer:
    """Build an Optimizer that includes torsal-direction terms.

    Default weights follow the paper's Table 2 (w_LC = 1, w_LCOrth = 0.01,
    w_torsal = 1, w_angle = 2). Note w_orth is 0.01 here against 2.0 in
    stage 2: once the congruence is established, holding the angle constraint
    hard would drag `l` off unit(cu x cv) instead of letting the radii move.

    `hinge` defaults to whatever stage 2 used.
    """
    if hinge is None:
        hinge = getattr(state, "hinge", True)
    bsp = state.bspline
    u, v = state.u_pts, state.v_pts
    r_uv  = state.r_uv
    sample = (state.u_sample_num, state.v_sample_num)
    n_sq  = (state.u_sample_num - 1) * (state.v_sample_num - 1)

    # Carry over previous LC solution. The paper flips any lines that ended up
    # on the wrong side of the normal "at the end of the first step".
    prev = state.lc_opt
    f_l, f_cp = prev.unpack("l", "rij")
    n_flat = state.n_surf.reshape(-1, 3)
    f_l = np.sign(np.einsum("ij,ij->i", f_l.reshape(-1, 3), n_flat))[:, None] * f_l.reshape(-1, 3)

    opt = Optimizer()
    opt.add_variable("rij",  f_cp)
    opt.add_variable("l",    f_l.ravel())
    if not hinge:
        opt.add_variable("mu", prev.unpack("mu"))
    opt.add_variable("nt1",  np.zeros(3 * n_sq))
    opt.add_variable("nt2",  np.zeros(3 * n_sq))
    opt.add_variable("u1",   np.zeros(n_sq))
    opt.add_variable("v1",   np.zeros(n_sq))
    opt.add_variable("u2",   np.zeros(n_sq))
    opt.add_variable("v2",   np.zeros(n_sq))
    if not hinge:
        opt.add_variable("theta", np.zeros(n_sq))

    lc_term = LineCong()
    opt.add_objective_term(lc_term, (bsp, r_uv, u, v), w=w_lc, ce=True)

    orth_term = LineCongOrth()
    opt.add_objective_term(orth_term, (bsp, r_uv, u, v, state.angle_lc, hinge),
                            w=w_orth, ce=True)

    torsal_term = Torsal()
    opt.add_objective_term(torsal_term,
                            (bsp, u, v, state.n_surf, sample),
                            w=w_torsal, ce=True)

    tangle_term = TorsalAngle()
    opt.add_objective_term(tangle_term, (state.angle_torsal, hinge),
                            w=w_tangle, ce=True)

    opt.unitize_variable("l",   3, w=10.0)
    opt.unitize_variable("nt1", 3, w=10.0)
    opt.unitize_variable("nt2", 3, w=10.0)
    opt.control_variable("nt1", 0.05)
    opt.control_variable("nt2", 0.05)
    opt.set_fairness("l", _vertex_adj(state), dim=3, w=0.01)

    opt.initialize_optimizer(verbose=True, adaptive_mu=False)
    state.torsal_opt = opt
    state.hinge = hinge
    return opt


def recompute_torsal_directions(state: MoebiusState) -> None:
    """Re-solve the analytic torsal directions from the current l and write them back.

    Paper Sec. 5.1, "Implementation details": the torsal directions and the
    plane normals n*_q are recomputed after a defined number of steps, which
    "prevents the optimization from getting stuck with degenerated torsal
    directions and has proven to yield faster convergence".

    Resets u1, v1, u2, v2, nt1, nt2 from the current l, then re-seeds the
    TorsalAngle slack and the frozen normalisers so everything is consistent.
    """
    opt   = state.torsal_opt
    terms = opt.objective_terms
    tors  = terms.get("Torsal")
    if tors is None:
        raise RuntimeError("No Torsal term — call setup_torsal_optimizer first.")

    l = opt.X[opt.var_idx["l"]].reshape(state.u_sample_num, state.v_sample_num, 3)
    tors._init_torsal_vars(opt.X, opt.var_idx, l)

    # nt1/nt2 moved, so the TorsalAngle slack and every frozen normaliser
    # (and the step-control anchors) must be re-seeded at the new iterate.
    ta = terms.get("TorsalAngle")
    if ta is not None and not getattr(ta, "hinge", True):
        nt1 = opt.X[opt.var_idx["nt1"]].reshape(-1, 3)
        nt2 = opt.X[opt.var_idx["nt2"]].reshape(-1, 3)
        dot = np.einsum("ij,ij->i", nt1, nt2)
        theta0 = np.sqrt(np.maximum(ta.cos2a - dot ** 2, 0.0))
        opt.X[opt.var_idx["theta"]] = np.maximum(theta0, 1e-3)
    for t in terms.values():
        t.accept_step(opt.X)


def run_torsal_optimizer(state: MoebiusState, max_iter: int = 100,
                         recompute_steps: Optional[int] = 10,
                         final_plain_steps: int = 0) -> None:
    """Run the torsal optimisation, recomputing the torsal directions periodically.

    Parameters
    ----------
    recompute_steps   recompute the analytic torsal directions every N steps
                      (paper Tables 1-2 call this RS, with values 10 and 25).
                      None disables it, reproducing the old single-shot
                      behaviour.
    final_plain_steps if > 0, run this many closing iterations with
                      w_angle = w_fair = 0, as the paper does "to improve the
                      correctness of the torsal directions".
    """
    opt = state.torsal_opt
    main_iter = max(max_iter - final_plain_steps, 0)

    # NOTE: hanan's Optimizer.optimize(max_iter=N) loops `while self.it < N`,
    # where self.it persists across calls. It is a TOTAL iteration budget, not
    # an increment, so a staged loop must pass cumulative targets.
    start = opt.it
    if not recompute_steps or recompute_steps >= main_iter:
        opt.optimize(max_iter=start + main_iter)
    else:
        done = 0
        while done < main_iter and not opt.stop:
            chunk = min(recompute_steps, main_iter - done)
            before = opt.it
            opt.optimize(max_iter=opt.it + chunk)
            if opt.it == before:          # converged or stalled
                break
            done += opt.it - before
            if done < main_iter:
                recompute_torsal_directions(state)

    if final_plain_steps > 0:
        # Only w_angle. Chakana's fairness terms already damp their own weight
        # to zero after `damp_iteration` steps, which is the behaviour we want.
        zeroed = {}
        for nm, t in opt.objective_terms.items():
            if nm == "TorsalAngle":
                zeroed[nm] = t.w
                t.set_weigth(0.0)
        print(f"Final {final_plain_steps} steps with w_angle = 0 "
              f"({', '.join(zeroed)})")
        opt.stop = False
        opt.optimize(max_iter=opt.it + final_plain_steps)
        for nm, w in zeroed.items():
            opt.objective_terms[nm].set_weigth(w)

    state.r_uv[2] = opt.unpack("rij")
    print(opt.get_report())


# ── Stage 4: remeshing ────────────────────────────────────────────────────────
#
# The external tool is `quadRemesher` (see QuadRemesher/Quad_Remesher/README.md).
# It takes a TRIANGLE mesh plus a frame field (two directions per element) and
# produces a polygon mesh — mostly quads — whose edges follow the field.
#
# Two ways to run it, both exposed here through run_remesher(..., batch=...):
#   batch=True   no window; integrate, extract, write <name>_Remeshed.obj, exit.
#   batch=False  opens the viewer: '+'/'-' change the gradient (quad density),
#                'R' previews the quad mesh, 'S' extracts and saves it.
#
# Our torsal field lives per QUAD of the sampling grid, so the exporter
# triangulates the grid and gives both triangles of a quad that quad's
# directions (--defined_on faces).

REMESHER_CANDIDATES = (
    PROJECT_ROOT.parent / "QuadRemesher" / "bin" / "quadRemesher",
    PROJECT_ROOT.parent / "QuadRemesher" / "Quad_Remesher" / "build" / "quadRemesher",
)


def resolve_remesher_bin(explicit: Optional[str] = None) -> str:
    """Locate the quadRemesher binary.

    Order: `explicit` argument, $REMESHER_BIN, the paths in
    REMESHER_CANDIDATES, then `quadRemesher` on $PATH.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Remesher binary not found: {explicit}")
        return explicit

    env = os.environ.get("REMESHER_BIN")
    if env and os.path.isfile(env):
        return env

    for cand in REMESHER_CANDIDATES:
        if cand.is_file():
            return str(cand)

    found = shutil.which("quadRemesher")
    if found:
        return found

    tried = "\n  ".join([str(c) for c in REMESHER_CANDIDATES] + ["$REMESHER_BIN", "$PATH"])
    raise FileNotFoundError(
        "Could not find the quadRemesher binary. Tried:\n  " + tried +
        "\nSet state.remesher_bin, or export REMESHER_BIN=/path/to/quadRemesher."
    )


def remesher_help(binary: Optional[str] = None) -> str:
    """Return the binary's --help text (used to check which CLI it has)."""
    binary = resolve_remesher_bin(binary)
    out = subprocess.run([binary, "--help"], capture_output=True, text=True)
    return out.stdout + out.stderr


def check_remesher(binary: Optional[str] = None) -> str:
    """Verify the binary understands the current flag-based CLI.

    Older builds took positional arguments (`quadRemesher mesh.obj D1 D2 ...`)
    and had no --gradient flag. Rebuild if this raises.
    """
    binary = resolve_remesher_bin(binary)
    help_text = remesher_help(binary)
    missing = [f for f in ("--field", "--gradient", "--d1", "--batch")
               if f not in help_text]
    if missing:
        raise RuntimeError(
            f"{binary} does not support {', '.join(missing)} — it is an older "
            "build with the positional CLI. Rebuild it:\n"
            "  cmake --build <QuadRemesher>/Quad_Remesher/build --target quadRemesher -j8"
        )
    print(f"Remesher OK: {binary}")
    return binary


# ── frame-field export ────────────────────────────────────────────────────────

def torsal_field_per_quad(state: MoebiusState):
    """Unit torsal directions (t1, t2) per quad of the sampling grid.

    The optimiser stores the torsal directions in the (du, dv) basis of each
    quad — du = v2 - v0, dv = v1 - v3 on the surface, matching energies.Torsal:

        t_k = u_k · du + v_k · dv

    Returns (t1, t2), both (n_quads, 3) and unit length.
    """
    if state.torsal_opt is None:
        raise RuntimeError("Run the torsal optimisation (stage 3) first.")

    u1, v1, u2, v2 = state.torsal_opt.unpack("u1", "v1", "u2", "v2")
    V, F = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)

    du = V[F[:, 2]] - V[F[:, 0]]
    dv = V[F[:, 1]] - V[F[:, 3]]

    t1 = u1[:, None] * du + v1[:, None] * dv
    t2 = u2[:, None] * du + v2[:, None] * dv

    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12
    return t1, t2


def export_frame_field(state: MoebiusState, basename: str = "remesh_input",
                       one_frame_per_quad: bool = True,
                       project_to_faces: bool = True) -> dict:
    """Write the triangulated surface + per-face frame field for quadRemesher.

    Produces, in state.save_path:
      <basename>.obj  triangle mesh (the remesher rejects quads)
      D1.dat, D2.dat  one unit direction per constrained triangle, '--format vec'
      indices.dat     which triangles those rows belong to (one_frame_per_quad)

    Each quad of the sampling grid becomes two triangles (hanan's
    triangulate_quads order: quad q -> triangles 2q and 2q+1).

    A torsal direction lives in the plane of its quad's two diagonals. When the
    quad is non-planar, neither triangle lies in that plane, and the remesher's
    in-plane projection rotates the direction — badly, where the grid is coarse
    relative to the curvature.

    `one_frame_per_quad` (default True) therefore constrains only ONE triangle
    per quad — the one whose plane needs the smaller rotation — and lets the
    remesher interpolate the field over the rest, which is what `--indices` is
    for. This both halves the rotation error and halves the constraint count,
    which tends to give a smoother field with fewer singularities. Set it False
    to constrain both triangles of every quad.

    `project_to_faces` (default True) applies the in-plane projection here
    rather than leaving it to the remesher. It does not change the result — it
    makes the written file equal to what is actually used, and lets us report
    the rotation.

    Returns the paths written plus element counts and diagnostics.
    """
    os.makedirs(state.save_path, exist_ok=True)

    V, F_quad = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    F_tri = np.array(triangulate_quads(F_quad.tolist()), dtype=np.int32)

    t1, t2 = torsal_field_per_quad(state)
    n_quads = len(F_quad)
    if len(t1) != n_quads:
        raise RuntimeError(
            f"Field/quad count mismatch: {len(t1)} directions, {n_quads} quads "
            "— the sampling grid changed after the torsal optimisation."
        )

    # quad q -> triangles 2q, 2q+1
    t1_tri = np.repeat(t1, 2, axis=0)
    t2_tri = np.repeat(t2, 2, axis=0)
    assert len(t1_tri) == len(F_tri), (len(t1_tri), len(F_tri))

    # Triangle normals, and how far each direction is out of its triangle's plane
    nrm = np.cross(V[F_tri[:, 1]] - V[F_tri[:, 0]],
                   V[F_tri[:, 2]] - V[F_tri[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    p1, rot1 = _project_into_plane(t1_tri, nrm)
    p2, rot2 = _project_into_plane(t2_tri, nrm)
    rot = np.maximum(rot1, rot2)

    # Keep, per quad, the triangle that needs the smaller rotation
    tri_idx = None
    if one_frame_per_quad:
        pick = np.argmin(rot.reshape(n_quads, 2), axis=1)
        tri_idx = 2 * np.arange(n_quads, dtype=np.int32) + pick.astype(np.int32)
        t1_tri, t2_tri = t1_tri[tri_idx], t2_tri[tri_idx]
        p1, p2, rot = p1[tri_idx], p2[tri_idx], rot[tri_idx]

    if project_to_faces:
        t1_tri, t2_tri = p1, p2

    obj_path = os.path.join(state.save_path, basename + ".obj")
    d1_path  = os.path.join(state.save_path, "D1.dat")
    d2_path  = os.path.join(state.save_path, "D2.dat")
    idx_path = os.path.join(state.save_path, "indices.dat") if tri_idx is not None else None

    write_obj(obj_path, V, F_tri)
    _write_vec_file(d1_path, t1_tri, "D1 - first torsal direction")
    _write_vec_file(d2_path, t2_tri, "D2 - second torsal direction")
    if idx_path is not None:
        with open(idx_path, "w") as f:
            f.write(" ".join(str(int(i)) for i in tri_idx) + "\n")

    # Angle between the two directions: the field is degenerate where they meet
    ang = np.degrees(np.arccos(np.clip(np.abs(
        np.einsum("ij,ij->i", t1, t2)), -1, 1)))

    print(f"Exported {len(V)} vertices, {len(F_tri)} triangles -> {state.save_path}")
    print(f"  {len(t1_tri)} frames on "
          + (f"1 of the 2 triangles of each of {n_quads} quads"
             if one_frame_per_quad else f"all {len(F_tri)} triangles"))
    print(f"  |angle(t1,t2)|  min {ang.min():.1f} deg  mean {ang.mean():.1f} deg  "
          f"max {ang.max():.1f} deg")
    if ang.min() < 5.0:
        print(f"  WARNING: {int((ang < 5).sum())} quads have near-parallel "
              "torsal directions - the field is degenerate there.")

    print(f"  in-plane rotation on the constrained triangles: "
          f"mean {rot.mean():.1f} deg, max {rot.max():.1f} deg")
    n_bad = int((rot > 15).sum())
    if n_bad:
        print(f"  WARNING: >15 deg on {n_bad} of {len(rot)} constrained "
              "triangles - the sampling quads are too non-planar there. Raise "
              "u_sample_num/v_sample_num, or lower --constraints.")

    paths = {"obj": obj_path, "d1": d1_path, "d2": d2_path}
    if idx_path is not None:
        paths["indices"] = idx_path
    state.frame_field_paths = paths

    return {**paths,
            "n_vertices": len(V), "n_triangles": len(F_tri),
            "n_frames": len(t1_tri), "n_quads": n_quads,
            "angle_t1_t2_deg": {"min": float(ang.min()), "mean": float(ang.mean()),
                                 "max": float(ang.max())},
            "rotation_deg": {"mean": float(rot.mean()), "max": float(rot.max()),
                              "n_over_15": n_bad}}


def _project_into_plane(d: np.ndarray, n: np.ndarray):
    """Project unit directions `d` into the planes with unit normals `n`.

    Returns (projected unit directions, rotation angle in degrees). Directions
    that are (nearly) parallel to the normal have no well-defined projection
    and are left unchanged.
    """
    q = d - np.einsum("ij,ij->i", d, n)[:, None] * n
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    degenerate = (nrm < 1e-8).ravel()
    q = np.where(degenerate[:, None], d, q / np.maximum(nrm, 1e-12))
    rot = np.degrees(np.arccos(np.clip(np.abs(
        np.einsum("ij,ij->i", d, q)), 0, 1)))
    return q, rot


def _write_vec_file(path: str, vecs: np.ndarray, comment: str) -> None:
    """Write one 'vx vy vz' per line ('--format vec')."""
    with open(path, "w") as f:
        f.write(f"# {comment}\n")
        for v in vecs:
            f.write(f"{v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")


def save_remesh_input(state: MoebiusState) -> str:
    """Export the frame field and pickle the state (stage-4 entry point)."""
    paths = export_frame_field(state)
    _save_state(state, "pre_remesh_state.pickle")
    return paths["obj"]


# ── running the remesher ──────────────────────────────────────────────────────

def remesher_command(state: MoebiusState,
                     gradient: float = 40.0,
                     batch: bool = True,
                     constraints: float = 1.0,
                     extractor: str = "qex",
                     name: Optional[str] = None,
                     binary: Optional[str] = None,
                     extra_args=()) -> list:
    """Build the quadRemesher argument list (without running it).

    Parameters
    ----------
    gradient    integration gradient size; HIGHER = DENSER quads (default 40).
                Relative to the bounding box, so comparable across meshes.
                On the reference mesh, 20/40/80 gave roughly 181/793/3364 quads.
    constraints fraction of field-carrying faces kept as hard constraints
                (0 < f <= 1). Lower values (0.1) let the solver smooth the
                field and usually give fewer singularities and cleaner quads.
    extractor   'qex' (robust at seams/singularities) or 'grid'.
    batch       True = headless; False = open the viewer ('+'/'-' gradient,
                'R' preview, 'S' save).
    """
    binary = resolve_remesher_bin(binary or state.remesher_bin)
    paths  = getattr(state, "frame_field_paths", None)
    if not paths:
        raise RuntimeError("Call export_frame_field(state) first.")
    if not 0.0 < constraints <= 1.0:
        raise ValueError(f"constraints must be in (0, 1], got {constraints}")
    if gradient <= 0:
        raise ValueError(f"gradient must be > 0, got {gradient}")

    cmd = [binary, paths["obj"],
           "--field", "data",
           "--d1", paths["d1"],
           "--d2", paths["d2"],
           "--format", "vec",
           "--defined_on", "faces",]
    if paths.get("indices"):
        cmd += ["--indices", paths["indices"]]
    cmd += [
           "--gradient", f"{gradient:g}",
           "--constraints", f"{constraints:g}",
           "--extractor", extractor,
           "--out", state.save_path,
           "--name", name or state.name]
    if batch:
        cmd.append("--batch")
    cmd.extend(extra_args)
    return cmd


def run_remesher(state: MoebiusState,
                 gradient: float = 40.0,
                 batch: bool = True,
                 constraints: float = 1.0,
                 extractor: str = "qex",
                 name: Optional[str] = None,
                 binary: Optional[str] = None,
                 load: bool = True,
                 verbose: bool = True,
                 extra_args=()) -> dict:
    """Run quadRemesher on the exported frame field.

    With batch=True it writes <save_path>/<name>_Remeshed.obj and (if
    load=True) loads it into the state. With batch=False the viewer opens and
    this blocks until you close it; press 'S' there to write the same file.

    Returns {'returncode', 'out_obj', 'gradient', 'stdout', 'stderr'}.
    """
    cmd = remesher_command(state, gradient=gradient, batch=batch,
                           constraints=constraints, extractor=extractor,
                           name=name, binary=binary, extra_args=extra_args)
    out_obj = os.path.join(state.save_path, (name or state.name) + "_Remeshed.obj")

    if verbose:
        print(" ".join(cmd), "\n")

    if batch:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout, stderr = proc.stdout, proc.stderr
        rc = proc.returncode
        if verbose and stdout:
            print(stdout.strip()[-2000:])
    else:
        # Viewer needs the terminal; don't capture.
        rc = subprocess.call(cmd)
        stdout = stderr = ""

    if rc != 0:
        print(f"Remesher failed (exit {rc}).")
        if stderr:
            print(stderr.strip()[-2000:])
        return {"returncode": rc, "out_obj": None, "gradient": gradient,
                "stdout": stdout, "stderr": stderr}

    state.remesh_gradient    = gradient
    state.remesh_constraints = constraints

    if load:
        if os.path.isfile(out_obj):
            load_remeshed_surface(state, out_obj)
        else:
            print(f"Remesher succeeded but {out_obj} is missing "
                  "(in the viewer, press 'S' to save).")

    return {"returncode": rc, "out_obj": out_obj, "gradient": gradient,
            "stdout": stdout, "stderr": stderr}


# ── loading the result ────────────────────────────────────────────────────────

def load_remeshed_surface(state: MoebiusState, path: Optional[str] = None) -> None:
    """Load the remeshed polygon OBJ and transfer the line congruence onto it.

    The extractor writes mostly quads plus a few other polygons, so
    state.F_remesh is a LIST of index lists (ragged). state.F_remesh_quads is
    an (nf, 4) array when every face is a quad and None otherwise - stage 5
    needs that array.
    """
    if path is None:
        path = os.path.join(state.save_path, state.name + "_Remeshed.obj")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Remeshed mesh not found: {path}")

    V, F_list = read_obj(path)
    state.V_remesh = V
    state.F_remesh = F_list

    sizes, counts = np.unique([len(f) for f in F_list], return_counts=True)
    if len(sizes) == 1 and sizes[0] == 4:
        state.F_remesh_quads = np.array(F_list, dtype=np.int32)
    else:
        state.F_remesh_quads = None

    # Transfer l from the sampling grid to the new vertices
    V_orig, F_quad = sample_bspline_surface(state.bspline, state.u_pts, state.v_pts)
    F_tri = np.array(triangulate_quads(F_quad.tolist()), dtype=np.int32)
    l_flat = state.torsal_opt.unpack("l").reshape(-1, 3)
    state.l_remesh = interpolate_lc(V, V_orig, F_tri, l_flat)

    print(f"Loaded {path}")
    print(f"  {len(V)} vertices, {len(F_list)} faces  "
          + ", ".join(f"{c}x{s}-gon" for s, c in zip(sizes, counts)))
    if state.F_remesh_quads is None:
        n_quad = int(counts[sizes == 4].sum()) if 4 in sizes else 0
        print(f"  NOTE: {len(F_list) - n_quad} non-quad faces - fine to view, but "
              "stage 5 (post-optimisation) needs an all-quad mesh. Adjust "
              "--gradient/--constraints, or use extractor='qex'.")


def remesh_quality(state: MoebiusState) -> dict:
    """Face-count / face-size summary of the remeshed mesh, for gradient tuning."""
    if state.F_remesh is None:
        raise RuntimeError("No remeshed mesh loaded.")
    V, F = state.V_remesh, state.F_remesh
    sizes = np.array([len(f) for f in F])
    # Rough face size: mean distance of face vertices to their centroid
    diam = []
    for f in F:
        p = V[f]
        diam.append(np.linalg.norm(p - p.mean(axis=0), axis=1).mean())
    diam = np.array(diam)
    info = {
        "gradient":    getattr(state, "remesh_gradient", None),
        "n_vertices":  len(V),
        "n_faces":     len(F),
        "n_quads":     int((sizes == 4).sum()),
        "quad_frac":   float((sizes == 4).mean()),
        "face_sizes":  {int(s): int(c) for s, c in
                        zip(*np.unique(sizes, return_counts=True))},
        "mean_radius": float(diam.mean()),
    }
    print(f"gradient {info['gradient']}  ->  {info['n_faces']} faces "
          f"({info['n_quads']} quads, {100*info['quad_frac']:.1f}%), "
          f"{info['n_vertices']} vertices, mean face radius {info['mean_radius']:.4f}")
    return info


def sweep_gradient(state: MoebiusState, gradients, **kwargs) -> list:
    """Run the remesher once per gradient value and report the face counts.

    Use this to pick a density, then re-run run_remesher() at the value you
    want so the state holds that mesh. Each run overwrites the OBJ unless you
    pass distinct `name`s, so the state ends up holding the LAST gradient.
    """
    rows = []
    for g in gradients:
        print(f"\n-- gradient {g} ------------------------------")
        res = run_remesher(state, gradient=g, batch=True, verbose=False, **kwargs)
        if res["returncode"] == 0 and state.F_remesh is not None:
            rows.append(remesh_quality(state))
        else:
            print(f"  failed (exit {res['returncode']})")
            rows.append({"gradient": g, "n_faces": None})
    return rows


# ── Stage 5: post-optimisation ────────────────────────────────────────────────

def setup_postopt_optimizer(state: MoebiusState,
                             w_sphere: float = 1.0,
                             w_support: float = 1e-1,
                             w_reg: float = 1e-2) -> Optimizer:
    """Build the post-optimisation Optimizer (sphere fit + support + regularity)."""
    V = state.V_remesh
    F = state.F_remesh_quads
    if F is None:
        raise RuntimeError(
            "Post-optimisation needs an all-quad remeshed mesh, but the "
            "extractor produced other polygons too (see load_remeshed_surface "
            "output). Re-run run_remesher() with a different --gradient / "
            "--constraints, or handle the n-gons first."
        )

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
    opt.initialize_optimizer(verbose=True, adaptive_mu=False)
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
                   state.V_remesh, state.F_remesh)   # write_obj handles n-gons

    # Post-opt mesh (if available)
    if state.postopt is not None:
        V_opt = state.postopt.unpack("v").reshape(-1, 3)
        write_obj(os.path.join(state.save_path, "optimised_mesh.obj"),
                   V_opt, state.F_remesh)

        from moebius.glyphs import sphere_centers_radii_from_params, spherical_panels_from_mesh
        A, B, C = state.postopt.unpack("A", "B", "C")
        B = B.reshape(-1, 3)
        centers, radii = sphere_centers_radii_from_params(A, B, C)
        pV, pF = spherical_panels_from_mesh(V_opt, state.F_remesh_quads,
                                             centers, radii)
        write_obj(os.path.join(state.save_path, "sphere_panels.obj"), pV, pF)

    _save_state(state, "state.pickle")
    print(f"Results saved to {state.save_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

# Attributes that hanan's ObjectiveTerm caches on first use and that cannot be
# pickled: jit-compiled jax closures built inside compute_grad_jax(). They are a
# runtime cache, rebuilt on the next call, so dropping them before dumping is
# safe. Chakana does the same for Optimizer._executor in its __getstate__; the
# proper fix is an ObjectiveTerm.__getstate__ upstream (see CHECKLIST.md).
_JAX_CACHE_ATTRS = ("_jax_jac_func", "_jax_func_jit", "_jax_dims")


def _drop_jax_caches(state: MoebiusState) -> None:
    """Remove unpicklable jax JIT caches from every objective term."""
    for attr in ("bspline_opt", "lc_opt", "torsal_opt", "postopt"):
        opt = getattr(state, attr, None)
        terms = getattr(opt, "objective_terms", None) if opt is not None else None
        if not terms:
            continue
        for term in terms.values():
            for cached in _JAX_CACHE_ATTRS:
                if hasattr(term, cached):
                    delattr(term, cached)


def _save_state(state: MoebiusState, filename: str) -> None:
    os.makedirs(state.save_path, exist_ok=True)
    path = os.path.join(state.save_path, filename)
    _drop_jax_caches(state)
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
