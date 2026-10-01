"""
Mean-curvature sign enforcement on the B-spline surface.

Penalises grid points where H(u,v) is too close to zero or has the wrong sign:

    E = Σ max(0, thresh - sign_H · H(p))²

where sign_H is the dominant sign of H on the surface (determined at init).
When all residuals vanish, H > thresh everywhere — the sphere-congruence
initialisation r = 1/H is then well-conditioned throughout the domain.

Variables: "cp_surf"  — B-spline surface control points (nu_cp × nv_cp × 3, flat).
Jacobian : finite differences (bspline evaluation uses splipy/numpy, not JAX).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm
from moebius.utils.bsplines import bspline_curvatures


class MeanCurvatureBspline(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name           = "MeanCurvatureBspline"
        self.jacobianMethod = "FD"

    def initialize_objective(self, X, var_idx, bsp, u_pts, v_pts,
                              H_thresh: float = 0.05):
        """
        Parameters
        ----------
        bsp      : splipy Surface object (will be mutated during optimisation).
        u_pts, v_pts : 1-D parameter arrays for the evaluation grid.
        H_thresh : minimum required value of sign_H · H at every grid point.
                   Increase if r = 1/H values are too large; decrease to be lenient.
        """
        self._cp_idx   = var_idx["cp_surf"]
        self._bsp      = bsp
        self._cp_shape = bsp.controlpoints.shape   # (nu_cp, nv_cp, 3)
        self._u_pts    = u_pts
        self._v_pts    = v_pts
        self._thresh   = H_thresh

        # Dominant H sign from current surface
        _, H0, _ = bspline_curvatures(bsp, u_pts, v_pts)
        median_H = float(np.median(H0.ravel()))
        self._sign_H = 1.0 if median_H >= 0 else -1.0

        self.num_residuals = len(u_pts) * len(v_pts)

    def res(self, X) -> np.ndarray:
        cp = X[self._cp_idx].reshape(self._cp_shape)
        self._bsp.controlpoints = cp
        _, H, _ = bspline_curvatures(self._bsp, self._u_pts, self._v_pts)
        # Positive residual when  sign_H · H < thresh  (H too close to 0 or wrong sign)
        return np.maximum(0.0, self._thresh - self._sign_H * H.ravel())
