"""
Torsal-plane angle constraint:

    E = Σ [(nt1 · nt2)² - cos²(α) + θ²]²

Forces the angle between the two torsal plane normals to be ≥ α degrees.
θ is a slack variable.

Variables: nt1, nt2, theta.
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class TorsalAngle(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "TorsalAngle"

    def initialize_objective(self, X, var_idx, angle_deg):
        self._var_nt1   = var_idx["nt1"]
        self._var_nt2   = var_idx["nt2"]
        self._var_theta = var_idx["theta"]

        F           = len(var_idx["nt1"]) // 3
        self._F     = F
        self.cos2a  = np.cos(np.deg2rad(angle_deg)) ** 2
        self.num_residuals = F

        r_idx = np.arange(F)
        self._rows = np.concatenate([
            np.repeat(r_idx, 3),   # nt1 (3 per residual)
            np.repeat(r_idx, 3),   # nt2
            r_idx,                 # theta (1 per residual)
        ]).astype(np.int32)
        self._cols = np.concatenate([
            var_idx["nt1"],
            var_idx["nt2"],
            var_idx["theta"],
        ]).astype(np.int32)

    def res(self, X) -> np.ndarray:
        nt1   = X[self._var_nt1].reshape(-1, 3)
        nt2   = X[self._var_nt2].reshape(-1, 3)
        theta = X[self._var_theta]
        dot   = np.einsum("ij,ij->i", nt1, nt2)
        return dot**2 - self.cos2a + theta**2

    def grad(self, X) -> np.ndarray:
        nt1   = X[self._var_nt1].reshape(-1, 3)
        nt2   = X[self._var_nt2].reshape(-1, 3)
        theta = X[self._var_theta]
        dot   = np.einsum("ij,ij->i", nt1, nt2)

        d_nt1   = (2 * dot[:, None] * nt2).ravel()   # ∂/∂nt1 = 2(nt1·nt2)*nt2
        d_nt2   = (2 * dot[:, None] * nt1).ravel()
        d_theta = 2 * theta
        return np.concatenate([d_nt1, d_nt2, d_theta])
