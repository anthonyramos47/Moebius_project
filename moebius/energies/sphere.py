"""
Sphere-congruence fitting energy (post-optimisation).

For each sphere face f with vertices {v_k}, the algebraic sphere equation is

    E_f = Σ_k [ A_f ‖v_k‖² - B_f · v_k + C_f ]²

where  A_f = 1/(2r_f),  B_f = c_f/r_f²,  C_f = ‖c_f‖²/(2r_f²) - r_f/2
are the "generalised sphere" parameters; c_f is the sphere centre, r_f its radius.

Jacobian: analytical.
Variables: v (mesh vertices), A (scalar/face), B (3-vector/face), C (scalar/face).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class SphereFit(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "SphereFit"

    def initialize_objective(self, X, var_idx, vertex_sph, v_fixed=None):
        """
        Parameters
        ----------
        vertex_sph : (F, k) int array  — vertex indices belonging to each sphere face.
                     Can be a list of 1-D arrays if faces have variable valence.
        """
        # Flatten to one list of (face, vertex) pairs
        pairs = [(f, v) for f, row in enumerate(vertex_sph) for v in row]
        face_idx = np.array([p[0] for p in pairs], dtype=np.int32)
        vert_idx = np.array([p[1] for p in pairs], dtype=np.int32)
        n_res = len(pairs)

        self._face_idx = face_idx
        self._vert_idx = vert_idx
        # Warm-up phase: the paper optimises "only ... the spheres" first, so
        # the vertices are held constant and contribute no Jacobian columns.
        self._v_fixed = None if v_fixed is None else np.asarray(v_fixed, float)

        # Variable index slices
        # With v_fixed the warm-up optimiser has no "v" variable at all.
        self._vv    = var_idx["v"] if self._v_fixed is None else None
        self._vA    = var_idx["A"]
        self._vB    = var_idx["B"]
        self._vC    = var_idx["C"]
        self.num_residuals = n_res

        r_idx  = np.arange(n_res)
        v3_idx = np.repeat(r_idx, 3)

        # vertex columns: 3 per residual
        v_cols = (3 * vert_idx[:, None] + np.arange(3)).ravel()
        v_cols_global = var_idx["v"][v_cols] if self._v_fixed is None else None

        # A columns: 1 per residual
        A_cols = var_idx["A"][face_idx]

        # B columns: 3 per residual (each face has 3 B components)
        B_cols = (3 * face_idx[:, None] + np.arange(3)).ravel()
        B_cols_global = var_idx["B"][B_cols]

        # C columns: 1 per residual
        C_cols = var_idx["C"][face_idx]

        if self._v_fixed is None:
            self._rows = np.concatenate([v3_idx, r_idx, v3_idx, r_idx]).astype(np.int32)
            self._cols = np.concatenate([v_cols_global, A_cols,
                                         B_cols_global, C_cols]).astype(np.int32)
        else:
            self._rows = np.concatenate([r_idx, v3_idx, r_idx]).astype(np.int32)
            self._cols = np.concatenate([A_cols, B_cols_global, C_cols]).astype(np.int32)

    def _v(self, X):
        if self._v_fixed is not None:
            return self._v_fixed.reshape(-1, 3)[self._vert_idx]
        return X[self._vv].reshape(-1, 3)[self._vert_idx]

    def res(self, X) -> np.ndarray:
        v = self._v(X)                                   # (n_res, 3)
        A = X[self._vA][self._face_idx]                   # (n_res,)
        B = X[self._vB].reshape(-1, 3)[self._face_idx]    # (n_res, 3)
        C = X[self._vC][self._face_idx]                   # (n_res,)
        return A * np.einsum("ij,ij->i", v, v) - np.einsum("ij,ij->i", v, B) + C

    def grad(self, X) -> np.ndarray:
        v = self._v(X)
        A = X[self._vA][self._face_idx]
        B = X[self._vB].reshape(-1, 3)[self._face_idx]
        C = X[self._vC][self._face_idx]

        d_v = (2 * A[:, None] * v - B).ravel()            # ∂/∂v: 3N
        d_A = np.einsum("ij,ij->i", v, v)                 # ∂/∂A: N
        d_B = (-v).ravel()                                 # ∂/∂B: 3N
        d_C = np.ones(len(A))                              # ∂/∂C: N
        if self._v_fixed is not None:
            return np.concatenate([d_A, d_B, d_C])
        return np.concatenate([d_v, d_A, d_B, d_C])
