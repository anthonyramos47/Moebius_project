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

    def accept_step(self, X) -> None:
        """Refresh the frozen normalisers at the new iterate.

        The analytic Jacobian treats 1/‖cu‖ and 1/‖cv‖ as constants — it does
        not differentiate them w.r.t. rij. So they must stay fixed while the
        optimizer evaluates res/grad at one X, and be refreshed only once a
        step is accepted. (They used to be recomputed inside grad(), which left
        res() dividing by a different norm than grad().)
        """
        cu, cv = self._d_c_uv(X[self._var_rij])
        self.cu_norm = np.linalg.norm(cu, axis=2)
        self.cv_norm = np.linalg.norm(cv, axis=2)

    def grad(self, X) -> np.ndarray:
        cp   = X[self._var_rij]
        l    = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        cu, cv = self._d_c_uv(cp)
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
