"""
Line-congruence angle constraint:

    E = Σ [(l · n)² - cos²(α) - μ²]²

Forces the angle between l and the surface normal n to be α degrees.
The slack variable μ absorbs the difference, making the constraint soft.

Variables: l, mu.
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class LineCongOrth(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "LineCongOrth"

    def initialize_objective(self, X, var_idx, bsp, r_uv, u_pts, v_pts, angle_deg):
        self._var_l  = var_idx["l"]
        self._var_mu = var_idx["mu"]
        self.n       = bsp.normal(u_pts, v_pts)           # (nu, nv, 3) constant
        self.cos2a   = np.cos(np.deg2rad(angle_deg)) ** 2
        self.u_pts   = u_pts
        self.v_pts   = v_pts

        N = len(u_pts) * len(v_pts)
        self.num_residuals = N

        r_idx = np.arange(N)
        self._rows = np.concatenate([
            np.repeat(r_idx, 3),   # wrt l (3 components per residual)
            r_idx,                 # wrt mu (1 per residual)
        ]).astype(np.int32)
        self._cols = np.concatenate([
            var_idx["l"],
            var_idx["mu"],
        ]).astype(np.int32)

    def res(self, X) -> np.ndarray:
        l  = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        mu = X[self._var_mu]
        ln = np.einsum("ijk,ijk->ij", l, self.n).ravel()
        return ln**2 - self.cos2a - mu**2

    def grad(self, X) -> np.ndarray:
        l  = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        mu = X[self._var_mu]
        ln = np.einsum("ijk,ijk->ij", l, self.n)  # (nu, nv)

        # ∂E/∂l = 2*(l·n)*n
        d_l = (2 * ln[..., None] * self.n).ravel()   # (3N,)
        # ∂E/∂mu = -2*mu
        d_mu = -2 * mu                                # (N,)
        return np.concatenate([d_l, d_mu])
