"""
Support-structure planarity energy (post-optimisation).

For each "dual face" (polygon formed by sphere centres around a mesh vertex),
enforce that consecutive sphere centres c_i, c_{i+1} lie in a common plane:

    E = Σ_face Σ_{edges(i,j)} ‖ (c_j - c_i)/‖c_j-c_i‖ × n_d ‖²

where n_d is the auxiliary normal of the dual face.

Uses JAX autodiff (analytical Jacobian is very complex for polygonal faces).
Variables: nd (sphere-centre directions / dual normals), v (mesh vertices — not optimised here).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class SupportPlanarity(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name   = "SupportPlanarity"
        self.jacobianMethod = "jax"

    def initialize_objective(self, X, var_idx, sph_sph_adj, inner_v):
        """
        Parameters
        ----------
        sph_sph_adj : list of lists — for each dual face, the sphere centre indices.
        inner_v     : 1-D int array — global vertex indices of inner dual vertices.
        """
        self._var_nd = var_idx["nd"]
        self._var_v  = var_idx["v"]

        self._sph_adj = [np.array(f, dtype=np.int32)
                         for f in sph_sph_adj if len(f) >= 3]

        # Precompute edge pairs per dual face
        self._edges = []
        n_res = 0
        for f in self._sph_adj:
            k = len(f)
            for i in range(k):
                self._edges.append((f[i], f[(i + 1) % k]))
            n_res += k

        self._inner_v = inner_v
        self.num_residuals = n_res

    def res(self, X) -> np.ndarray:
        nd = X[self._var_nd].reshape(-1, 3)
        v  = X[self._var_v ].reshape(-1, 3)
        r  = []
        edge_idx = 0
        for fi, f in enumerate(self._sph_adj):
            n_d  = nd[self._inner_v[fi]]
            for i in range(len(f)):
                ci = v[f[i]]
                cj = v[f[(i + 1) % len(f)]]
                diff = cj - ci
                norm_diff = diff / (np.linalg.norm(diff) + 1e-12)
                cross = np.cross(norm_diff, n_d)
                r.append(np.dot(cross, cross))
        return np.array(r)
