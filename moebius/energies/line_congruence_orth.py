"""
Line-congruence angle constraint (paper Sec. 5.1, E_LCorth).

Enforces a good angle between the congruence line l and the surface normal n:

    <l, n>^2 >= cos^2(a)        i.e.   angle(l, n) <= a

Two equivalent formulations:

`hinge=True` (default) — one-sided residual, no slack variable:

    r = max(0, cos^2(a) - <l, n>^2)

`hinge=False` — the published form with a dummy variable mu:

    r = <l, n>^2 - cos^2(a) - mu^2

The square on <l, n> is deliberate (paper): it makes the term blind to the
orientation of l, so flipped lines cost nothing and are flipped back at the end
of the first step.

Why the hinge: dr/dmu = -2 mu, so mu = 0 is a stationary point with a null
Jacobian column and mu can never leave zero. Measured on Tunel/CMC, mu sat at
exactly 0 on >90% of grid points, i.e. hundreds of singular directions in the
Gauss-Newton system. The hinge has no such variable. Its residual has a kink at
the constraint boundary, but the ENERGY sum r^2 = max(0, ...)^2 is C^1 there
(the gradient 2 r dr/dX -> 0 as r -> 0), so LM is well behaved.

Variables: l  (plus mu when hinge=False).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class LineCongOrth(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "LineCongOrth"

    def initialize_objective(self, X, var_idx, bsp, r_uv, u_pts, v_pts,
                             angle_deg, hinge: bool = True):
        self._var_l  = var_idx["l"]
        self.hinge   = bool(hinge)
        self._var_mu = None if self.hinge else var_idx["mu"]
        self.n       = bsp.normal(u_pts, v_pts)           # (nu, nv, 3) constant
        self.cos2a   = np.cos(np.deg2rad(angle_deg)) ** 2
        self.u_pts   = u_pts
        self.v_pts   = v_pts

        N = len(u_pts) * len(v_pts)
        self.num_residuals = N

        r_idx = np.arange(N)
        if self.hinge:
            self._rows = np.repeat(r_idx, 3).astype(np.int32)
            self._cols = np.asarray(var_idx["l"], dtype=np.int32)
        else:
            self._rows = np.concatenate([
                np.repeat(r_idx, 3),   # wrt l (3 components per residual)
                r_idx,                 # wrt mu (1 per residual)
            ]).astype(np.int32)
            self._cols = np.concatenate([
                var_idx["l"],
                var_idx["mu"],
            ]).astype(np.int32)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ln(self, X):
        """<l, n> on the grid, shape (nu, nv)."""
        l = X[self._var_l].reshape(len(self.u_pts), len(self.v_pts), 3)
        return np.einsum("ijk,ijk->ij", l, self.n)

    # ── residual / Jacobian ──────────────────────────────────────────────────

    def res(self, X) -> np.ndarray:
        ln2 = self._ln(X) ** 2
        if self.hinge:
            return np.maximum(self.cos2a - ln2, 0.0).ravel()
        mu = X[self._var_mu]
        return ln2.ravel() - self.cos2a - mu ** 2

    def grad(self, X) -> np.ndarray:
        ln = self._ln(X)

        if self.hinge:
            # r = max(0, cos^2 a - <l,n>^2);  active where <l,n>^2 < cos^2 a
            # dr/dl = -2 <l,n> n   on active rows, 0 otherwise
            active = (ln ** 2 < self.cos2a)
            d_l = (-2.0 * (ln * active)[..., None] * self.n).ravel()
            return d_l

        # smooth form: r = <l,n>^2 - cos^2 a - mu^2
        d_l  = (2 * ln[..., None] * self.n).ravel()   # (3N,)
        d_mu = -2 * X[self._var_mu]                   # (N,)
        return np.concatenate([d_l, d_mu])
