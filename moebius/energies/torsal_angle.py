"""
Torsal-plane angle constraint (paper Sec. 5.1, E_angle).

The two torsal planes meet along lc; the angle between them is the angle
between their normals nt1, nt2. We want that intersection angle not to deviate
too far from pi/2, i.e.

    angle(nt1, nt2) >= a        i.e.   <nt1, nt2>^2 <= cos^2(a)

so a LARGER a is the stronger requirement (a = 90 deg would demand exactly
orthogonal torsal planes). Useful range is 45..90 deg, 45 being the minimum.

Two equivalent formulations:

`hinge=True` (default) — one-sided residual, no slack variable:

    r = max(0, <nt1, nt2>^2 - cos^2(a))

`hinge=False` — the published form with a dummy variable nu (here `theta`):

    r = <nt1, nt2>^2 - cos^2(a) + theta^2

Why the hinge: dr/dtheta = 2 theta, so theta = 0 is a stationary point with a
null Jacobian column. Started at zero it stayed at zero on EVERY face, which
silently turned the inequality into the equality angle(nt1,nt2) = a — dragging
well-separated torsal planes back down to the threshold. Measured on CMC:
mean <nt1,nt2>^2 = 0.7452 against a target of exactly 0.7500, mean angle
29.6 deg with a = 30 deg, 77% of faces within +-5 deg of the threshold.

The hinge is also better conditioned here than for LineCongOrth: a violating
face has |<nt1,nt2>| > cos(a) >= cos(45 deg), so ||dr/dnt|| >= 1.41 and never
vanishes where it is needed.

Variables: nt1, nt2  (plus theta when hinge=False).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class TorsalAngle(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "TorsalAngle"

    def initialize_objective(self, X, var_idx, angle_deg, hinge: bool = True):
        self._var_nt1 = var_idx["nt1"]
        self._var_nt2 = var_idx["nt2"]
        self.hinge    = bool(hinge)
        self._var_theta = None if self.hinge else var_idx["theta"]

        F           = len(var_idx["nt1"]) // 3
        self._F     = F
        self.cos2a  = np.cos(np.deg2rad(angle_deg)) ** 2
        self.num_residuals = F

        r_idx = np.arange(F)
        if self.hinge:
            self._rows = np.concatenate([
                np.repeat(r_idx, 3),   # nt1
                np.repeat(r_idx, 3),   # nt2
            ]).astype(np.int32)
            self._cols = np.concatenate([
                var_idx["nt1"],
                var_idx["nt2"],
            ]).astype(np.int32)
        else:
            # Seed the slack off its zero-gradient point, or it never moves.
            nt1 = X[var_idx["nt1"]].reshape(-1, 3)
            nt2 = X[var_idx["nt2"]].reshape(-1, 3)
            dot = np.einsum("ij,ij->i", nt1, nt2)
            theta0 = np.sqrt(np.maximum(self.cos2a - dot ** 2, 0.0))
            X[var_idx["theta"]] = np.maximum(theta0, 1e-3)

            self._rows = np.concatenate([
                np.repeat(r_idx, 3),   # nt1
                np.repeat(r_idx, 3),   # nt2
                r_idx,                 # theta
            ]).astype(np.int32)
            self._cols = np.concatenate([
                var_idx["nt1"],
                var_idx["nt2"],
                var_idx["theta"],
            ]).astype(np.int32)

    # ── residual / Jacobian ──────────────────────────────────────────────────

    def _dot(self, X):
        nt1 = X[self._var_nt1].reshape(-1, 3)
        nt2 = X[self._var_nt2].reshape(-1, 3)
        return nt1, nt2, np.einsum("ij,ij->i", nt1, nt2)

    def res(self, X) -> np.ndarray:
        _, _, dot = self._dot(X)
        if self.hinge:
            return np.maximum(dot ** 2 - self.cos2a, 0.0)
        return dot ** 2 - self.cos2a + X[self._var_theta] ** 2

    def grad(self, X) -> np.ndarray:
        nt1, nt2, dot = self._dot(X)

        if self.hinge:
            # r = max(0, <nt1,nt2>^2 - cos^2 a);  active where it exceeds
            act     = (dot ** 2 > self.cos2a)
            dot_a   = dot * act
            d_nt1   = (2 * dot_a[:, None] * nt2).ravel()
            d_nt2   = (2 * dot_a[:, None] * nt1).ravel()
            return np.concatenate([d_nt1, d_nt2])

        d_nt1   = (2 * dot[:, None] * nt2).ravel()   # d/dnt1 = 2(nt1.nt2) nt2
        d_nt2   = (2 * dot[:, None] * nt1).ravel()
        d_theta = 2 * X[self._var_theta]
        return np.concatenate([d_nt1, d_nt2, d_theta])
