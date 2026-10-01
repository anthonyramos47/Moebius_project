"""
Face-regularity energy (post-optimisation).

For each interior edge (f_i, f_j) shared by two sphere faces, enforce that
the sphere parameters are compatible:

    E = ‖(A_j B_i - A_i B_j) · (w - u)/‖w - u‖‖²

where (u, w) are the two endpoint vertices of the edge.

Uses JAX autodiff.
Variables: v (mesh vertices), A (scalar/face), B (3-vector/face).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class RegFaces(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "RegFaces"
        self.jacobianMethod = "jax"

    def initialize_objective(self, X, var_idx, e_f_f, e_v_v):
        """
        Parameters
        ----------
        e_f_f : (2, E) int  — face index pairs for each interior edge.
        e_v_v : (2, E) int  — vertex index pairs for each interior edge.
        """
        self._fi = np.array(e_f_f[0], dtype=np.int32)
        self._fj = np.array(e_f_f[1], dtype=np.int32)
        self._ui = np.array(e_v_v[0], dtype=np.int32)
        self._wi = np.array(e_v_v[1], dtype=np.int32)

        self._vv = var_idx["v"]
        self._vA = var_idx["A"]
        self._vB = var_idx["B"]

        # Precompute fixed edge lengths (updated each step via accept_step)
        u = X[var_idx["v"]].reshape(-1, 3)[self._ui]
        w = X[var_idx["v"]].reshape(-1, 3)[self._wi]
        self._wu_norm = np.linalg.norm(w - u, axis=1) + 1e-12
        self.num_residuals = len(self._fi)

    def accept_step(self, X) -> None:
        u = X[self._vv].reshape(-1, 3)[self._ui]
        w = X[self._vv].reshape(-1, 3)[self._wi]
        self._wu_norm = np.linalg.norm(w - u, axis=1) + 1e-12

    def res(self, X) -> np.ndarray:
        v = X[self._vv].reshape(-1, 3)
        A = X[self._vA]
        B = X[self._vB].reshape(-1, 3)

        u   = v[self._ui];  w  = v[self._wi]
        A_i = A[self._fi];  A_j = A[self._fj]
        B_i = B[self._fi];  B_j = B[self._fj]

        wu    = w - u
        wu_n  = wu / self._wu_norm[:, None]
        cij   = A_j[:, None] * B_i - A_i[:, None] * B_j
        return np.einsum("ij,ij->i", cij, wu_n)
