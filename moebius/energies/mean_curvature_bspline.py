"""
Mean-curvature conditioning on the B-spline surface.

Keeps sign_H · H inside a band, as two one-sided (hinge) residual blocks:

    E = Σ max(0, H_thresh - sign_H·H(p))²          floor, |H| not too small
      + w_max · Σ max(0, sign_H·H(p) - H_max)²     ceiling, |H| not too large

where sign_H is the dominant sign of H on the surface (determined at init).

The floor is what makes the sphere-congruence initialisation r = 1/H
well-conditioned: when it vanishes, |H| > H_thresh everywhere.

The ceiling exists because the floor alone is satisfied most cheaply by a
*local* bump — at a near-flat point, denting the surface raises |H| there with
almost no change elsewhere, and the result reads as a crease. Nothing in the
floor prefers a broad deformation to a sharp one. `set_lap_smooth` resists it
only until its weight damps to zero (10 iterations by default), which is well
before a 30-iteration run ends, so the penalty has to live here to be active
throughout. Capping sign_H·H also bounds the radii from below, keeping the
congruence inside 1/H_max <= r <= 1/H_thresh rather than letting a crease
produce arbitrarily small spheres.

Both blocks are in one term rather than two because the Jacobian is finite
differences: each control-point column costs a full curvature evaluation over
the grid, so a second term would double the cost of the stage to apply a
penalty that reuses exactly the same H.

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
                              H_thresh: float = 0.05,
                              H_max: float | None = None,
                              w_max: float = 1.0):
        """
        Parameters
        ----------
        bsp      : splipy Surface object (will be mutated during optimisation).
        u_pts, v_pts : 1-D parameter arrays for the evaluation grid.
        H_thresh : minimum required value of sign_H · H at every grid point.
                   Increase if r = 1/H values are too large; decrease to be lenient.
        H_max    : maximum allowed value of sign_H · H, or None for no ceiling.
                   This is what penalises creases. It is an absolute curvature,
                   so scale it to the surface: the input is normalised to a
                   bounding box of 2, and a sensible starting point is a small
                   multiple of the surface's own initial max |H|.
        w_max    : weight of the ceiling block *relative to the floor*. The
                   optimiser's own `w` scales the whole term, so the ceiling's
                   effective weight is w · w_max; the residual is scaled by
                   sqrt(w_max) to make that exact.
        """
        self._cp_idx   = var_idx["cp_surf"]
        self._bsp      = bsp
        self._cp_shape = bsp.controlpoints.shape   # (nu_cp, nv_cp, 3)
        self._u_pts    = u_pts
        self._v_pts    = v_pts
        self._thresh   = H_thresh
        self._H_max    = H_max
        self._max_scale = float(np.sqrt(w_max))

        # Dominant H sign from current surface
        _, H0, _ = bspline_curvatures(bsp, u_pts, v_pts)
        median_H = float(np.median(H0.ravel()))
        self._sign_H = 1.0 if median_H >= 0 else -1.0

        if H_max is not None and H_max <= H_thresh:
            raise ValueError(
                f"H_max ({H_max}) must exceed H_thresh ({H_thresh}); the two "
                "hinges would otherwise fight over every grid point and no "
                "surface could satisfy both.")

        n = len(u_pts) * len(v_pts)
        self.num_residuals = n if H_max is None else 2 * n

    def res(self, X) -> np.ndarray:
        cp = X[self._cp_idx].reshape(self._cp_shape)
        self._bsp.controlpoints = cp
        _, H, _ = bspline_curvatures(self._bsp, self._u_pts, self._v_pts)
        sH = self._sign_H * H.ravel()

        # Positive residual when  sign_H · H < thresh  (H too close to 0 or wrong sign)
        floor = np.maximum(0.0, self._thresh - sH)
        if self._H_max is None:
            return floor
        # Positive residual where the surface is sharper than allowed.
        ceiling = self._max_scale * np.maximum(0.0, sH - self._H_max)
        return np.concatenate([floor, ceiling])
