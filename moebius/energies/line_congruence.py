"""
Line-congruence energy  E = Σ (l · cu/‖cu‖)² + (l · cv/‖cv‖)²

Minimising this makes the line congruence l(u,v) orthogonal to the
sphere-congruence tangent vectors cu and cv, i.e. l is the normal of
the sphere-congruence surface c(u,v) = s(u,v) + r(u,v)·n(u,v).

Variables optimised: rij (B-spline control points of r(u,v)), l (line congruence).
"""

import numpy as np
from scipy.interpolate import bisplev
from hanan.optimization.objective_term import ObjectiveTerm

from moebius.utils.bsplines import normal_derivatives_uv


class LineCong(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "LineCong"

    # ── initialisation ────────────────────────────────────────────────────────

    def initialize_objective(self, X, var_idx, bsp, r_uv, u_pts, v_pts):
        self.u_pts = u_pts
        self.v_pts = v_pts
        self.r_bs  = r_uv          # mutable list [t_u, t_v, c, k_u, k_v]

        # Precompute constant surface quantities
        self.su = bsp.derivative(u_pts, v_pts, d=(1, 0))
        self.sv = bsp.derivative(u_pts, v_pts, d=(0, 1))
        self.n  = bsp.normal(u_pts, v_pts)
        self.nu, self.nv = normal_derivatives_uv(bsp, u_pts, v_pts)

        N       = len(u_pts) * len(v_pts)
        n_cp    = len(r_uv[2])

        # Compute initial cu/cv norms for the first step
        cu, cv  = self._d_c_uv(r_uv[2])
        self.cu_norm = np.linalg.norm(cu, axis=2)
        self.cv_norm = np.linalg.norm(cv, axis=2)

        # Pre-compute ∂(cu)/∂a_k, ∂(cv)/∂a_k for each control point a_k
        I     = np.eye(n_cp)
        r_tmp = list(r_uv)
        self.d_a_cu = np.zeros((n_cp, len(u_pts), len(v_pts), 3))
        self.d_a_cv = np.zeros_like(self.d_a_cu)
        for k in range(n_cp):
            r_tmp[2] = I[k]
            da_r  = bisplev(u_pts, v_pts, r_tmp)
            da_ru = bisplev(u_pts, v_pts, r_tmp, dx=1, dy=0)
            da_rv = bisplev(u_pts, v_pts, r_tmp, dx=0, dy=1)
            self.d_a_cu[k] = da_ru[..., None] * self.n + da_r[..., None] * self.nu
            self.d_a_cv[k] = da_rv[..., None] * self.n + da_r[..., None] * self.nv

        self.num_residuals = 2 * N

        # ── sparsity structure ────────────────────────────────────────────────
        # E_cu block: rows [0, N)     E_cv block: rows [N, 2N)
        r_E_cu = np.arange(N)
        r_E_cv = np.arange(N, 2 * N)
        l_idx  = var_idx["l"]

        # wrt l  (3 columns per residual)
        rows_l_cu = np.repeat(r_E_cu, 3)
        rows_l_cv = np.repeat(r_E_cv, 3)
        cols_l    = np.tile(l_idx.reshape(N, 3), 1).reshape(-1)  # identical for both blocks
        # wrt l_idx is just var_idx["l"] in order [lx0,ly0,lz0, lx1,...]
        # which equals l_idx itself if laid out that way
        cols_l_cu = l_idx                            # shape (3N,)  already interleaved
        cols_l_cv = l_idx

        # wrt rij (one column per control point per residual)
        rij_idx = var_idx["rij"]
        rows_rij_cu = np.tile(r_E_cu, n_cp)          # [0..N-1, 0..N-1, ...]
        rows_rij_cv = np.tile(r_E_cv, n_cp)
        cols_rij    = np.repeat(rij_idx, N)           # [k0 repeated N, k1 repeated N, ...]

        self._rows = np.concatenate([
            rows_l_cu, rows_l_cv,
            rows_rij_cu, rows_rij_cv,
        ]).astype(np.int32)
        self._cols = np.concatenate([
            cols_l_cu, cols_l_cv,
            cols_rij,  cols_rij,
        ]).astype(np.int32)

    # ── per-iteration computation ─────────────────────────────────────────────

    def res(self, X) -> np.ndarray:
        cp = X[self._var_rij]
        l  = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        cu, cv = self._d_c_uv(cp)
        r = np.concatenate([
            np.einsum("ijk,ijk->ij", l, cu).ravel() / self.cu_norm.ravel(),
            np.einsum("ijk,ijk->ij", l, cv).ravel() / self.cv_norm.ravel(),
        ])
        return r

    def grad(self, X) -> np.ndarray:
        cp = X[self._var_rij]
        l  = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        cu, cv = self._d_c_uv(cp)

        self.cu_norm = np.linalg.norm(cu, axis=2)
        self.cv_norm = np.linalg.norm(cv, axis=2)

        N    = len(self.u_pts) * len(self.v_pts)
        n_cp = len(cp)

        # ∂/∂l  E_cu = cu/‖cu‖   (same shape as l)
        d_l_cu = (cu / self.cu_norm[..., None]).reshape(-1)  # (3N,)
        d_l_cv = (cv / self.cv_norm[..., None]).reshape(-1)

        # ∂/∂a_k  E_cu = l·(∂cu/∂a_k) / ‖cu‖
        d_rij_cu = np.stack([
            np.einsum("ijk,ijk->ij", self.d_a_cu[k], l).ravel() / self.cu_norm.ravel()
            for k in range(n_cp)
        ]).ravel()                                           # (n_cp * N,)

        d_rij_cv = np.stack([
            np.einsum("ijk,ijk->ij", self.d_a_cv[k], l).ravel() / self.cv_norm.ravel()
            for k in range(n_cp)
        ]).ravel()

        return np.concatenate([d_l_cu, d_l_cv, d_rij_cu, d_rij_cv])

    # ── called once after _initialize_objective stores var_idx ───────────────

    def initialize_objective(self, X, var_idx, bsp, r_uv, u_pts, v_pts):
        # store variable slices used in grad/res
        self._var_rij = var_idx["rij"]
        self._var_l   = var_idx["l"]
        # delegate to the actual init
        self._init_body(X, var_idx, bsp, r_uv, u_pts, v_pts)

    # rename the first definition to avoid Python overriding it
    _init_body = lambda self, *a, **kw: None   # overwritten below

    def _d_c_uv(self, cp):
        """Compute cu, cv from current control points cp."""
        self.r_bs[2] = cp
        r   = bisplev(self.u_pts, self.v_pts, self.r_bs)
        ru  = bisplev(self.u_pts, self.v_pts, self.r_bs, dx=1, dy=0)
        rv  = bisplev(self.u_pts, self.v_pts, self.r_bs, dx=0, dy=1)
        cu = self.su + ru[..., None] * self.n + r[..., None] * self.nu
        cv = self.sv + rv[..., None] * self.n + r[..., None] * self.nv
        return cu, cv


# Rewrite cleanly (Python method resolution order issue with two definitions):
del LineCong

class LineCong(ObjectiveTerm):
    """
    E = Σ (l · cu/‖cu‖)² + (l · cv/‖cv‖)²

    Enforces that the line congruence l(u,v) is normal to the sphere-congruence
    surface c(u,v) = s(u,v) + r(u,v)·n(u,v).
    """

    def __init__(self):
        super().__init__()
        self.name = "LineCong"

    def initialize_objective(self, X, var_idx, bsp, r_uv, u_pts, v_pts):
        self._var_rij = var_idx["rij"]
        self._var_l   = var_idx["l"]
        self.u_pts    = u_pts
        self.v_pts    = v_pts
        self.r_bs     = r_uv  # mutable list — we modify r_bs[2] in place

        self.su  = bsp.derivative(u_pts, v_pts, d=(1, 0))
        self.sv  = bsp.derivative(u_pts, v_pts, d=(0, 1))
        self.n   = bsp.normal(u_pts, v_pts)
        self.nu, self.nv = normal_derivatives_uv(bsp, u_pts, v_pts)

        N     = len(u_pts) * len(v_pts)
        n_cp  = len(r_uv[2])

        cu, cv = self._d_c_uv(r_uv[2])
        self.cu_norm = np.linalg.norm(cu, axis=2)
        self.cv_norm = np.linalg.norm(cv, axis=2)

        # ∂(cu)/∂a_k, ∂(cv)/∂a_k  — constant, precomputed once
        I     = np.eye(n_cp)
        r_tmp = list(r_uv)
        self.d_a_cu = np.zeros((n_cp, len(u_pts), len(v_pts), 3))
        self.d_a_cv = np.zeros_like(self.d_a_cu)
        for k in range(n_cp):
            r_tmp[2] = I[k]
            da_r  = bisplev(u_pts, v_pts, r_tmp)
            da_ru = bisplev(u_pts, v_pts, r_tmp, dx=1, dy=0)
            da_rv = bisplev(u_pts, v_pts, r_tmp, dx=0, dy=1)
            self.d_a_cu[k] = da_ru[..., None] * self.n + da_r[..., None] * self.nu
            self.d_a_cv[k] = da_rv[..., None] * self.n + da_r[..., None] * self.nv

        self.num_residuals = 2 * N

        # Sparsity: residuals [0,N) = E_cu,  [N,2N) = E_cv
        r_cu = np.arange(N)
        r_cv = np.arange(N, 2 * N)
        l_idx   = var_idx["l"]    # shape (3N,) interleaved [lx0,ly0,lz0,lx1,...]
        rij_idx = var_idx["rij"]  # shape (n_cp,)

        rows_l_cu  = np.repeat(r_cu, 3)
        rows_l_cv  = np.repeat(r_cv, 3)
        rows_rij   = np.tile(r_cu, n_cp)            # E_cu rows repeated per control point
        rows_rij_v = np.tile(r_cv, n_cp)            # E_cv rows

        self._rows = np.concatenate([
            rows_l_cu, rows_l_cv,
            rows_rij,  rows_rij_v,
        ]).astype(np.int32)
        self._cols = np.concatenate([
            l_idx, l_idx,                           # l cols same for E_cu and E_cv
            np.repeat(rij_idx, N),                  # each rij col repeated N times (E_cu)
            np.repeat(rij_idx, N),                  # same pattern for E_cv
        ]).astype(np.int32)

    def res(self, X) -> np.ndarray:
        cp = X[self._var_rij]
        l  = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        cu, cv = self._d_c_uv(cp)
        return np.concatenate([
            (np.einsum("ijk,ijk->ij", l, cu) / self.cu_norm).ravel(),
            (np.einsum("ijk,ijk->ij", l, cv) / self.cv_norm).ravel(),
        ])

    def grad(self, X) -> np.ndarray:
        cp   = X[self._var_rij]
        l    = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        cu, cv = self._d_c_uv(cp)
        self.cu_norm = np.linalg.norm(cu, axis=2)
        self.cv_norm = np.linalg.norm(cv, axis=2)
        n_cp = len(cp)

        d_l_cu   = (cu / self.cu_norm[..., None]).ravel()
        d_l_cv   = (cv / self.cv_norm[..., None]).ravel()
        d_rij_cu = np.concatenate([
            (np.einsum("ijk,ijk->ij", self.d_a_cu[k], l) / self.cu_norm).ravel()
            for k in range(n_cp)
        ])
        d_rij_cv = np.concatenate([
            (np.einsum("ijk,ijk->ij", self.d_a_cv[k], l) / self.cv_norm).ravel()
            for k in range(n_cp)
        ])
        return np.concatenate([d_l_cu, d_l_cv, d_rij_cu, d_rij_cv])

    def _d_c_uv(self, cp):
        self.r_bs[2] = cp
        r  = bisplev(self.u_pts, self.v_pts, self.r_bs)
        ru = bisplev(self.u_pts, self.v_pts, self.r_bs, dx=1, dy=0)
        rv = bisplev(self.u_pts, self.v_pts, self.r_bs, dx=0, dy=1)
        cu = self.su + ru[..., None] * self.n + r[..., None] * self.nu
        cv = self.sv + rv[..., None] * self.n + r[..., None] * self.nv
        return cu, cv
