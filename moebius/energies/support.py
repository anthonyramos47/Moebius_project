"""
Support-structure planarity energy (post-optimisation), paper Sec. 5.2 E_supp.

Around every inner vertex of the remeshed quad mesh the four sphere centres
form a face c_q of the "mesh of centres". That face must be planar for a
torsion-free node to exist. With an auxiliary unit normal n_cq per ring,

    psi(c_i, c_j) := <c_i - c_j, n_cq>,     E_supp = sum_q sum_{c_i c_j in c_q} psi^2

so n_cq is driven PERPENDICULAR to every edge of the ring, which is what makes
the ring planar.

Implemented in the POLYNOMIAL form used by the original QS_project/energies/
Support.py, which drops the division by A:

    A_p B_q - A_q B_p = 2 A_p A_q (c_q - c_p)

Same zero set, but writing it with centres c = B/(2A) would put 1/A and 1/A^2
into the Jacobian and blow up for large spheres (A -> 0). The scale factor
2 A_p A_q is absorbed by the weight.

Note this is a dot product, not a cross product. The previous version minimised
‖(c_j - c_i)/‖·‖ x n_d‖², which is zero when the normal is PARALLEL to the edge
— the opposite condition, and unsatisfiable for a ring with two non-parallel
edges. It also used numpy inside a jax-traced residual and so raised
TracerArrayConversionError on the first Jacobian evaluation, which stopped the
whole post-optimisation at iteration 0.

Sphere centres come from the algebraic sphere parameters as c_f = B_f / (2 A_f).

Jacobian: analytical.
Variables: A (scalar/face), B (3-vector/face), nd (one unit normal per ring).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class SupportPlanarity(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "SupportPlanarity"

    def initialize_objective(self, X, var_idx, sph_sph_adj):
        """
        Parameters
        ----------
        sph_sph_adj : list of lists — ordered face rings from Mesh.dual_top().
                      Entry i holds the face indices around the i-th inner
                      vertex; one normal `nd` per entry.
        """
        self._var_nd = var_idx["nd"]
        self._var_A  = var_idx["A"]
        self._var_B  = var_idx["B"]

        rings = [np.asarray(f, dtype=np.int32) for f in sph_sph_adj if len(f) >= 3]
        self._rings = rings

        # Flatten the ring edges once: residual k compares faces ci[k], cj[k]
        # using the normal of ring ring_of[k].
        ci, cj, ring_of = [], [], []
        for ri, f in enumerate(rings):
            k = len(f)
            for i in range(k):
                ci.append(f[i]); cj.append(f[(i + 1) % k]); ring_of.append(ri)
        self._ci      = np.asarray(ci, dtype=np.int32)
        self._cj      = np.asarray(cj, dtype=np.int32)
        self._ring_of = np.asarray(ring_of, dtype=np.int32)
        self.num_residuals = len(self._ci)

        n_res = self.num_residuals
        r_idx = np.arange(n_res)

        def flat3(idx):
            return np.stack([3 * idx + k for k in range(3)], axis=1).ravel()

        A_idx = np.asarray(var_idx["A"], dtype=np.int32)
        B_idx = np.asarray(var_idx["B"], dtype=np.int32)
        nd_idx = np.asarray(var_idx["nd"], dtype=np.int32)

        # columns: A_ci, A_cj (1 each), B_ci, B_cj, nd (3 each)
        self._rows = np.concatenate([
            r_idx, r_idx,
            np.repeat(r_idx, 3), np.repeat(r_idx, 3), np.repeat(r_idx, 3),
        ]).astype(np.int32)
        self._cols = np.concatenate([
            A_idx[self._ci], A_idx[self._cj],
            B_idx[flat3(self._ci)], B_idx[flat3(self._cj)],
            nd_idx[flat3(self._ring_of)],
        ]).astype(np.int32)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parts(self, X):
        A  = X[self._var_A]
        B  = X[self._var_B].reshape(-1, 3)
        nd = X[self._var_nd].reshape(-1, 3)
        return A, B, nd

    # ── residual / Jacobian ──────────────────────────────────────────────────

    def res(self, X) -> np.ndarray:
        A, B, nd = self._parts(X)
        d = A[self._ci][:, None] * B[self._cj] - A[self._cj][:, None] * B[self._ci]
        return np.einsum("ij,ij->i", d, nd[self._ring_of])

    def grad(self, X) -> np.ndarray:
        A, B, nd = self._parts(X)
        n = nd[self._ring_of]                        # (m, 3)
        Bi, Bj = B[self._ci], B[self._cj]
        Ai, Aj = A[self._ci], A[self._cj]

        # r = <A_i B_j - A_j B_i, n>
        #   dr/dA_i =  <B_j, n>      dr/dA_j = -<B_i, n>
        #   dr/dB_i = -A_j n         dr/dB_j =  A_i n
        #   dr/dn   =  A_i B_j - A_j B_i
        d_A_i = np.einsum("ij,ij->i", Bj, n)
        d_A_j = -np.einsum("ij,ij->i", Bi, n)
        d_B_i = (-Aj[:, None] * n).ravel()
        d_B_j = (Ai[:, None] * n).ravel()
        d_nd  = (Ai[:, None] * Bj - Aj[:, None] * Bi).ravel()

        return np.concatenate([d_A_i, d_A_j, d_B_i, d_B_j, d_nd])
