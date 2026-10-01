"""
Torsal-plane energy for the sphere-congruence line congruence.

Eight residual blocks per quad face F:
  [0,F)   lt_nt1  : (lt1/‖lt1‖) · nt1 = 0
  [F,2F)  t_nt1   : t1 · nt1 = 0
  [2F,3F) lt_nt2  : (lt2/‖lt2‖) · nt2 = 0
  [3F,4F) t_nt2   : t2 · nt2 = 0
  [4F,5F) lc_nt1  : (lc/‖lc‖) · nt1 = 0
  [5F,6F) lc_nt2  : (lc/‖lc‖) · nt2 = 0
  [6F,7F) t1_unit : ‖t1‖² - 1 = 0
  [7F,8F) t2_unit : ‖t2‖² - 1 = 0

where  t_k = u_k·du + v_k·dv,   lt_k = u_k·lu + v_k·lv,
       lc = (l[i0]+l[i1]+l[i2]+l[i3])/4,   lu = l[i2]-l[i0],   lv = l[i1]-l[i3].

Variables: l, u1, v1, u2, v2, nt1, nt2.
Initialisation also sets X[u1,v1,u2,v2,nt1,nt2] to analytic torsal directions.
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from moebius.utils.mesh_utils import (
    flat_array_variables, lc_info_at_grid_points, torsal_directions, unit, vec_dot
)


class Torsal(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "Torsal"

    # ── initialisation ────────────────────────────────────────────────────────

    def initialize_objective(self, X, var_idx, bsp, u_pts, v_pts, n, sample):
        nu, nv = sample
        F = (nu - 1) * (nv - 1)

        # Quad face vertex indices on the (nu × nv) grid
        indices = np.array([
            [nv * i + j, nv * i + j + 1, nv * (i + 1) + j + 1, nv * (i + 1) + j]
            for i in range(nu - 1) for j in range(nv - 1)
        ])
        self.i0, self.i1, self.i2, self.i3 = (indices[:, k] for k in range(4))
        self.i0f = flat_array_variables(self.i0)
        self.i1f = flat_array_variables(self.i1)
        self.i2f = flat_array_variables(self.i2)
        self.i3f = flat_array_variables(self.i3)

        # Surface tangents at face centres (constant)
        s_uv  = bsp(u_pts, v_pts).reshape(-1, 3)
        v0, v1, v2, v3 = (s_uv[self.i0], s_uv[self.i1],
                           s_uv[self.i2], s_uv[self.i3])
        self.du = v2 - v0
        self.dv = v1 - v3

        # Cache variable index slices
        self.vi = {k: var_idx[k] for k in ("l", "u1", "v1", "u2", "v2", "nt1", "nt2")}
        self.nu_sample = nu
        self.nv_sample = nv
        self._F = F

        # Initialise torsal variables in X
        l = X[var_idx["l"]].reshape(nu, nv, 3)
        n_bsp = bsp.normal(u_pts, v_pts)
        l = np.sign(np.einsum("ijk,ijk->ij", l, n_bsp))[..., None] * l
        self._init_torsal_vars(X, var_idx, l)

        # Running norms updated each iteration
        lt1 = (X[var_idx["u1"]][:, None] * self.du +
               X[var_idx["v1"]][:, None] * self.dv)
        lt2 = (X[var_idx["u2"]][:, None] * self.du +
               X[var_idx["v2"]][:, None] * self.dv)
        lc, _, _ = lc_info_at_grid_points(l)
        self.lt1_norm = np.linalg.norm(lt1, axis=1) + 1e-12
        self.lt2_norm = np.linalg.norm(lt2, axis=1) + 1e-12
        self.lc_norm  = np.linalg.norm(lc.reshape(-1, 3), axis=1) + 1e-12

        self.num_residuals = 8 * F
        self._build_sparsity(var_idx, F)

    def _init_torsal_vars(self, X, var_idx, l):
        F = self._F
        lc, lu, lv = lc_info_at_grid_points(l)
        lc = lc.reshape(-1, 3)
        lu = lu.reshape(-1, 3)
        lv = lv.reshape(-1, 3)
        lc_u = lc / (np.linalg.norm(lc, axis=1, keepdims=True) + 1e-12)

        t1, t2, ut1, vt1, ut2, vt2, _ = torsal_directions(lc_u, lu, lv, self.du, self.dv)
        nt1 = unit(np.cross(lc_u, t1))
        nt2 = unit(np.cross(lc_u, t2))

        # Write initial values into the optimizer variable vector
        for name, val in [("u1", ut1), ("v1", vt1), ("u2", ut2), ("v2", vt2),
                          ("nt1", nt1.ravel()), ("nt2", nt2.ravel())]:
            X[var_idx[name]] = val

    def _build_sparsity(self, var_idx, F):
        """Pre-compute _rows and _cols from variable index arrays."""
        r   = np.arange(F)
        r3  = np.repeat(r, 3)

        u1  = var_idx["u1"];   v1  = var_idx["v1"]
        u2  = var_idx["u2"];   v2  = var_idx["v2"]
        nt1 = var_idx["nt1"];  nt2 = var_idx["nt2"]
        l   = var_idx["l"]

        # lt_nt1 rows [0,F)
        R_lt1 = np.concatenate([r, r, r3, r3, r3, r3, r3])
        C_lt1 = np.concatenate([u1, v1,
                                 l[self.i0f], l[self.i2f],
                                 l[self.i1f], l[self.i3f],
                                 nt1])
        # t_nt1 rows [F,2F)
        R_tnt1 = np.concatenate([r + F, r + F, r3 + F])
        C_tnt1 = np.concatenate([u1, v1, nt1])

        # lt_nt2 rows [2F,3F)
        R_lt2 = np.concatenate([r + 2*F, r + 2*F, r3 + 2*F, r3 + 2*F,
                                 r3 + 2*F, r3 + 2*F, r3 + 2*F])
        C_lt2 = np.concatenate([u2, v2,
                                 l[self.i0f], l[self.i2f],
                                 l[self.i1f], l[self.i3f],
                                 nt2])
        # t_nt2 rows [3F,4F)
        R_tnt2 = np.concatenate([r + 3*F, r + 3*F, r3 + 3*F])
        C_tnt2 = np.concatenate([u2, v2, nt2])

        # lc_nt1 rows [4F,5F)
        R_lc1 = np.concatenate([r3 + 4*F, r3 + 4*F, r3 + 4*F, r3 + 4*F, r3 + 4*F])
        C_lc1 = np.concatenate([l[self.i0f], l[self.i1f], l[self.i2f], l[self.i3f], nt1])

        # lc_nt2 rows [5F,6F)
        R_lc2 = np.concatenate([r3 + 5*F, r3 + 5*F, r3 + 5*F, r3 + 5*F, r3 + 5*F])
        C_lc2 = np.concatenate([l[self.i0f], l[self.i1f], l[self.i2f], l[self.i3f], nt2])

        # t1_unit rows [6F,7F)
        R_tu1 = np.concatenate([r + 6*F, r + 6*F])
        C_tu1 = np.concatenate([u1, v1])

        # t2_unit rows [7F,8F)
        R_tu2 = np.concatenate([r + 7*F, r + 7*F])
        C_tu2 = np.concatenate([u2, v2])

        self._rows = np.concatenate([
            R_lt1, R_tnt1, R_lt2, R_tnt2,
            R_lc1, R_lc2, R_tu1, R_tu2,
        ]).astype(np.int32)
        self._cols = np.concatenate([
            C_lt1, C_tnt1, C_lt2, C_tnt2,
            C_lc1, C_lc2, C_tu1, C_tu2,
        ]).astype(np.int32)

    # ── per-iteration ─────────────────────────────────────────────────────────

    def _unpack(self, X):
        v = self.vi
        l   = X[v["l"]].reshape(self.nu_sample, self.nv_sample, 3)
        u1  = X[v["u1"]];  v1  = X[v["v1"]]
        u2  = X[v["u2"]];  v2  = X[v["v2"]]
        nt1 = X[v["nt1"]].reshape(-1, 3)
        nt2 = X[v["nt2"]].reshape(-1, 3)
        lc, lu, lv = lc_info_at_grid_points(l)
        return (l, u1, v1, u2, v2, nt1, nt2,
                lc.reshape(-1, 3), lu.reshape(-1, 3), lv.reshape(-1, 3))

    def res(self, X) -> np.ndarray:
        l, u1, v1, u2, v2, nt1, nt2, lc, lu, lv = self._unpack(X)
        F   = self._F
        t1  = u1[:, None] * self.du + v1[:, None] * self.dv
        t2  = u2[:, None] * self.du + v2[:, None] * self.dv
        lt1 = u1[:, None] * lu + v1[:, None] * lv
        lt2 = u2[:, None] * lu + v2[:, None] * lv
        lc_n = lc / (np.linalg.norm(lc, axis=1, keepdims=True) + 1e-12)
        lt1n = self.lt1_norm
        lt2n = self.lt2_norm

        r = np.empty(8 * F)
        r[0:F]   = vec_dot(lt1, nt1) / lt1n           # lt_nt1
        r[F:2*F] = vec_dot(t1,  nt1)                  # t_nt1
        r[2*F:3*F] = vec_dot(lt2, nt2) / lt2n         # lt_nt2
        r[3*F:4*F] = vec_dot(t2,  nt2)                # t_nt2
        r[4*F:5*F] = vec_dot(lc_n, nt1)               # lc_nt1
        r[5*F:6*F] = vec_dot(lc_n, nt2)               # lc_nt2
        r[6*F:7*F] = vec_dot(t1,  t1)  - 1            # t1_unit
        r[7*F:8*F] = vec_dot(t2,  t2)  - 1            # t2_unit
        return r

    def grad(self, X) -> np.ndarray:
        l, u1, v1, u2, v2, nt1, nt2, lc, lu, lv = self._unpack(X)
        F   = self._F
        t1  = u1[:, None] * self.du + v1[:, None] * self.dv
        t2  = u2[:, None] * self.du + v2[:, None] * self.dv
        lt1 = u1[:, None] * lu + v1[:, None] * lv
        lt2 = u2[:, None] * lu + v2[:, None] * lv

        lcn   = np.linalg.norm(lc, axis=1, keepdims=True) + 1e-12
        lc_u  = lc / lcn
        lt1n  = self.lt1_norm[:, None]
        lt2n  = self.lt2_norm[:, None]

        # Update running norms
        self.lt1_norm = np.linalg.norm(lt1, axis=1) + 1e-12
        self.lt2_norm = np.linalg.norm(lt2, axis=1) + 1e-12
        self.lc_norm  = np.linalg.norm(lc, axis=1)  + 1e-12

        # ── lt_nt1  [0,F) ──────────────────────────────────────────────────────
        # ∂/∂u1 = (lu·nt1)/‖lt1‖
        d_u1_lt1   = vec_dot(lu, nt1) / self.lt1_norm
        d_v1_lt1   = vec_dot(lv, nt1) / self.lt1_norm
        # ∂/∂l[i0] = -u1*nt1/‖lt1‖  (lu = l[i2]-l[i0])
        d_l0_lt1   = -(u1[:, None] * nt1 / lt1n).ravel()
        d_l2_lt1   =  (u1[:, None] * nt1 / lt1n).ravel()
        # ∂/∂l[i1] =  v1*nt1/‖lt1‖  (lv = l[i1]-l[i3])
        d_l1_lt1   =  (v1[:, None] * nt1 / lt1n).ravel()
        d_l3_lt1   = -(v1[:, None] * nt1 / lt1n).ravel()
        # ∂/∂nt1 = lt1/‖lt1‖
        d_nt1_lt1  = (lt1 / lt1n).ravel()

        # ── t_nt1  [F,2F) ─────────────────────────────────────────────────────
        d_u1_tnt1  = vec_dot(self.du, nt1)
        d_v1_tnt1  = vec_dot(self.dv, nt1)
        d_nt1_tnt1 = t1.ravel()

        # ── lt_nt2  [2F,3F) ───────────────────────────────────────────────────
        d_u2_lt2   = vec_dot(lu, nt2) / self.lt2_norm
        d_v2_lt2   = vec_dot(lv, nt2) / self.lt2_norm
        d_l0_lt2   = -(u2[:, None] * nt2 / lt2n).ravel()
        d_l2_lt2   =  (u2[:, None] * nt2 / lt2n).ravel()
        d_l1_lt2   =  (v2[:, None] * nt2 / lt2n).ravel()
        d_l3_lt2   = -(v2[:, None] * nt2 / lt2n).ravel()
        d_nt2_lt2  = (lt2 / lt2n).ravel()

        # ── t_nt2  [3F,4F) ────────────────────────────────────────────────────
        d_u2_tnt2  = vec_dot(self.du, nt2)
        d_v2_tnt2  = vec_dot(self.dv, nt2)
        d_nt2_tnt2 = t2.ravel()

        # ── lc_nt1  [4F,5F) ───────────────────────────────────────────────────
        # ∂/∂l[i] = nt1/(4*‖lc‖)  (lc = mean of l[i0..i3])
        d_li_lc1   = (nt1 / (4 * self.lc_norm[:, None])).ravel()
        d_nt1_lc1  = lc_u.ravel()

        # ── lc_nt2  [5F,6F) ───────────────────────────────────────────────────
        d_li_lc2   = (nt2 / (4 * self.lc_norm[:, None])).ravel()
        d_nt2_lc2  = lc_u.ravel()

        # ── t1_unit  [6F,7F) ──────────────────────────────────────────────────
        d_u1_tu1   = 2 * vec_dot(self.du, t1)
        d_v1_tu1   = 2 * vec_dot(self.dv, t1)

        # ── t2_unit  [7F,8F) ──────────────────────────────────────────────────
        d_u2_tu2   = 2 * vec_dot(self.du, t2)
        d_v2_tu2   = 2 * vec_dot(self.dv, t2)

        return np.concatenate([
            # lt_nt1: u1, v1, l[i0], l[i2], l[i1], l[i3], nt1
            d_u1_lt1, d_v1_lt1, d_l0_lt1, d_l2_lt1, d_l1_lt1, d_l3_lt1, d_nt1_lt1,
            # t_nt1: u1, v1, nt1
            d_u1_tnt1, d_v1_tnt1, d_nt1_tnt1,
            # lt_nt2: u2, v2, l[i0], l[i2], l[i1], l[i3], nt2
            d_u2_lt2, d_v2_lt2, d_l0_lt2, d_l2_lt2, d_l1_lt2, d_l3_lt2, d_nt2_lt2,
            # t_nt2: u2, v2, nt2
            d_u2_tnt2, d_v2_tnt2, d_nt2_tnt2,
            # lc_nt1: l[i0], l[i1], l[i2], l[i3], nt1
            d_li_lc1, d_li_lc1, d_li_lc1, d_li_lc1, d_nt1_lc1,
            # lc_nt2: l[i0], l[i1], l[i2], l[i3], nt2
            d_li_lc2, d_li_lc2, d_li_lc2, d_li_lc2, d_nt2_lc2,
            # t1_unit: u1, v1
            d_u1_tu1, d_v1_tu1,
            # t2_unit: u2, v2
            d_u2_tu2, d_v2_tu2,
        ])
