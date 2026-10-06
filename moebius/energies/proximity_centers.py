"""
Proximity of the sphere centres to the optimised sphere congruence
(paper Sec. 5.2, E_prox_C).

Stage 2/3 produce a sphere congruence whose centres form the surface
c(u,v) = s(u,v) + r(u,v) n(u,v). The post-optimisation must not drift away
from it, so with v_f the closest point of a centre on that surface and n_f the
unit normal there:

    E_prox_C = sum_f  <c_f - v_f, n_f>^2  +  eps (c_f - v_f)^2

Written POLYNOMIALLY. The centres are c = B/(2A), so

    c - v_f = (B - 2 A v_f) / (2A)

and the residuals use the numerator B - 2 A v_f directly: same zero set, no
1/A or 1/A^2 in the Jacobian (A -> 0 is a plane, i.e. a very large sphere,
which is a perfectly ordinary thing to want here).

    E_T:  r = <B - 2 A v_f, n_f>         dr/dB = n_f    dr/dA = -2 <v_f, n_f>
    E_D:  r = eps (B - 2 A v_f)          dr/dB = eps    dr/dA = -2 eps v_f

QS_project/energies/Proximity_C.py had the derivatives of the polynomial form
but set the residual to (c - v_f)*eps, i.e. the two differed by a factor
1/(2A). It also refreshed its reference normal inside compute(); here the
closest point and its normal are frozen per step and refreshed in
accept_step(), so res and grad always agree at one X.

Jacobian: analytical.
Variables: A, B.
"""

import numpy as np
import igl
from hanan.optimization.objective_term import ObjectiveTerm


class ProximityCenters(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "ProximityCenters"

    def initialize_objective(self, X, var_idx, V_ref, F_ref, epsilon: float = 0.1):
        self._var_A = np.asarray(var_idx["A"], dtype=np.int32)
        self._var_B = np.asarray(var_idx["B"], dtype=np.int32)
        self.V_ref  = np.asarray(V_ref, dtype=float)
        self.F_ref  = np.asarray(F_ref, dtype=np.int32)
        self.eps    = float(epsilon)

        nf = len(self._var_A)
        self._nf = nf
        self._project(X)

        # rows: [0, nf)   tangential          [nf, 4nf)  distance (3 per face)
        self.num_residuals = 4 * nf
        r = np.arange(nf)

        def f3(idx):
            return np.stack([3 * idx + k for k in range(3)], axis=1).ravel()

        B_all = self._var_B
        A_rep = np.repeat(self._var_A, 3)

        self._rows = np.concatenate([
            np.repeat(r, 3),              # E_T wrt B
            r,                            # E_T wrt A
            np.arange(nf, 4 * nf),        # E_D wrt B (one row per component)
            np.arange(nf, 4 * nf),        # E_D wrt A
        ]).astype(np.int32)
        self._cols = np.concatenate([
            B_all, self._var_A, B_all, A_rep,
        ]).astype(np.int32)

    # ── frozen closest point / normal ────────────────────────────────────────

    def _centres(self, X):
        A = X[self._var_A]
        B = X[self._var_B].reshape(-1, 3)
        return A, B, B / (2.0 * A[:, None] + 1e-12)

    def _project(self, X):
        _, _, c = self._centres(X)
        _, _, proj = igl.point_mesh_squared_distance(c, self.V_ref, self.F_ref)
        self._vf = np.asarray(proj, dtype=float)
        d = c - self._vf
        n = np.linalg.norm(d, axis=1, keepdims=True)
        # Where the centre already sits on the surface the offset gives no
        # direction; keep the previous normal rather than a zero row.
        prev = getattr(self, "_nf_vec", None)
        self._nf_vec = np.where(n > 1e-10, d / np.maximum(n, 1e-12),
                                prev if prev is not None else np.array([0., 0., 1.]))

    def accept_step(self, X) -> None:
        self._project(X)

    # ── residual / Jacobian ──────────────────────────────────────────────────

    def res(self, X) -> np.ndarray:
        A, B, _ = self._centres(X)
        poly = B - 2.0 * A[:, None] * self._vf          # (nf, 3)
        out = np.empty(4 * self._nf)
        out[:self._nf] = np.einsum("ij,ij->i", poly, self._nf_vec)
        out[self._nf:] = (self.eps * poly).ravel()
        return out

    def grad(self, X) -> np.ndarray:
        nf_vec = self._nf_vec
        d_T_B = nf_vec.ravel()
        d_T_A = -2.0 * np.einsum("ij,ij->i", self._vf, nf_vec)
        d_D_B = np.full(3 * self._nf, self.eps)
        d_D_A = (-2.0 * self.eps * self._vf).ravel()
        return np.concatenate([d_T_B, d_T_A, d_D_B, d_D_A])
