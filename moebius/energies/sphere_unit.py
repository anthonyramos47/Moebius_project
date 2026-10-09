"""
Sphere-coefficient normalisation (post-optimisation), paper Sec. 5.2 E_unit.

    phi(f) := <B_f, B_f> - 4 A_f C_f - 1,      E_unit = sum_f phi^2

The algebraic sphere psi(x) = A x^2 - <B, x> + C is homogeneous: scaling
(A, B, C) by any factor describes the same sphere, and (0, 0, 0) satisfies
E_sphere *exactly*. Without this term the post-optimisation can drive every
sphere coefficient to zero — a global minimum that encodes no sphere at all.
The paper states it "is used to normalize the coefficients of the sphere
equation and prevents the coefficients from vanishing".

With this normalisation the sphere centre and radius are
    c = B / (2A),    r^2 = (<B,B> - 4AC) / (4A^2) = 1 / (4A^2).

Jacobian: analytical.
Variables: A, B, C (one sphere per face).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class SphereUnit(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "SphereUnit"

    def initialize_objective(self, X, var_idx):
        self._var_A = np.asarray(var_idx["A"], dtype=np.int32)
        self._var_B = np.asarray(var_idx["B"], dtype=np.int32)
        self._var_C = np.asarray(var_idx["C"], dtype=np.int32)

        nf = len(self._var_A)
        self.num_residuals = nf
        r_idx = np.arange(nf)

        self._rows = np.concatenate([
            r_idx,                  # A
            np.repeat(r_idx, 3),    # B
            r_idx,                  # C
        ]).astype(np.int32)
        self._cols = np.concatenate([
            self._var_A, self._var_B, self._var_C,
        ]).astype(np.int32)

    def res(self, X) -> np.ndarray:
        A = X[self._var_A]
        B = X[self._var_B].reshape(-1, 3)
        C = X[self._var_C]
        return np.einsum("ij,ij->i", B, B) - 4.0 * A * C - 1.0

    def grad(self, X) -> np.ndarray:
        A = X[self._var_A]
        B = X[self._var_B].reshape(-1, 3)
        C = X[self._var_C]
        return np.concatenate([
            -4.0 * C,            # d/dA
            (2.0 * B).ravel(),   # d/dB
            -4.0 * A,            # d/dC
        ])
