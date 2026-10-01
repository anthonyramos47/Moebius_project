"""
Support-structure planarity energy (post-optimisation).

For each dual face (ordered ring of quad-mesh faces around an inner vertex),
the sphere centres {c_f = B_f/(2*A_f)} should be coplanar:

    E = Σ_{inner v} Σ_{edge (fi,fj) in ring} ‖ (c_fj - c_fi)/‖·‖ × n_d ‖²

where n_d is the auxiliary dual-face normal variable.

Uses JAX autodiff.
Variables: A (scalar/face), B (3-vector/face), nd (dual normals, one per inner vertex).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class SupportPlanarity(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name           = "SupportPlanarity"
        self.jacobianMethod = "jax"

    def initialize_objective(self, X, var_idx, sph_sph_adj):
        """
        Parameters
        ----------
        sph_sph_adj : list of lists — ordered face rings from Mesh.dual_top().
                      Entry i contains the face indices around the i-th inner vertex.
                      One nd vector per entry.
        """
        self._var_nd = var_idx["nd"]
        self._var_A  = var_idx["A"]
        self._var_B  = var_idx["B"]

        self._sph_adj = [np.array(f, dtype=np.int32)
                         for f in sph_sph_adj if len(f) >= 3]

        n_res = sum(len(f) for f in self._sph_adj)
        self.num_residuals = n_res

    def res(self, X) -> np.ndarray:
        nd = X[self._var_nd].reshape(-1, 3)
        A  = X[self._var_A]                      # (nf,)
        B  = X[self._var_B].reshape(-1, 3)       # (nf, 3)
        # Sphere centres: c_f = B_f / (2 * A_f)
        c  = B / (2 * A[:, None] + 1e-12)

        r = []
        for fi, f in enumerate(self._sph_adj):
            n_d = nd[fi]                          # dual normal for this ring
            k   = len(f)
            for i in range(k):
                ci   = c[f[i]]
                cj   = c[f[(i + 1) % k]]
                diff = cj - ci
                diff_n = diff / (np.linalg.norm(diff) + 1e-12)
                cross  = np.cross(diff_n, n_d)
                r.append(np.dot(cross, cross))
        return np.array(r)
