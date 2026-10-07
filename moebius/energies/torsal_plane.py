"""
Discrete torsal-plane energy (post-optimisation), paper Sec. 5.2 E_torsal plane.

For a support structure to exist, the two node axes l_u, l_w at the ends of a
mesh edge uw must be coplanar with the edge itself. With an auxiliary unit
normal n_uw per edge,

    E = sum_uw  <l_u, n_uw>^2 + <l_w, n_uw>^2 + <(w-u)/||w-u||, n_uw>^2

with the constraint ||n_uw|| = 1 (add it with Optimizer.unitize_variable).

Ported from QS_project/energies/Tor_Planarity.py (now only in git history)
with two corrections:

* The residual divided by ||w-u|| but d/dw and d/du did not, so the Jacobian
  did not match its own residual. Here ||w-u|| is a FROZEN normaliser applied
  consistently to the residual and to every derivative.
* It refreshed ||w-u|| at the end of compute(), i.e. inside the Jacobian
  evaluation. That is now accept_step(), so res and grad stay consistent while
  the optimizer evaluates them at one X.

The paper initialises n_uw as the normalised cross product of u - w with
l_u + l_w, the AVERAGE of the two line directions (see `init_edge_normals`);
the original used l_u alone, which biases the plane toward one endpoint.

Jacobian: analytical.
Variables: v (vertices), l (node axis per vertex), n_l (unit normal per edge).
"""

import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


def init_edge_normals(v, l, ui, wi):
    """n_uw = normalise( (u - w) x (l_u + l_w) ), paper Sec. 5.2."""
    u, w = v[ui], v[wi]
    n = np.cross(u - w, l[ui] + l[wi])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    # Degenerate only if the edge is parallel to the averaged line direction;
    # fall back to any vector orthogonal to the edge.
    bad = (nn[:, 0] < 1e-12)
    if bad.any():
        e = u[bad] - w[bad]
        alt = np.tile(np.array([1.0, 0.0, 0.0]), (int(bad.sum()), 1))
        flip = np.abs(e[:, 0]) > 0.9
        alt[flip] = np.array([0.0, 1.0, 0.0])
        n[bad] = np.cross(e, alt)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(nn, 1e-12)


class TorsalPlane(ObjectiveTerm):

    def __init__(self):
        super().__init__()
        self.name = "TorsalPlane"

    def initialize_objective(self, X, var_idx, e_v_v):
        self._ui = np.asarray(e_v_v[0], dtype=np.int32)
        self._wi = np.asarray(e_v_v[1], dtype=np.int32)
        E = len(self._ui)
        self._E = E

        self._var_v = np.asarray(var_idx["v"],   dtype=np.int32)
        self._var_l = np.asarray(var_idx["l"],   dtype=np.int32)
        self._var_n = np.asarray(var_idx["n_l"], dtype=np.int32)

        v = X[self._var_v].reshape(-1, 3)
        self._e_norm = np.linalg.norm(v[self._wi] - v[self._ui], axis=1) + 1e-12

        self.num_residuals = 3 * E
        r = np.arange(E)

        def f3(idx):
            return np.stack([3 * idx + k for k in range(3)], axis=1).ravel()

        vi_u, vi_w = self._var_v[f3(self._ui)], self._var_v[f3(self._wi)]
        li_u, li_w = self._var_l[f3(self._ui)], self._var_l[f3(self._wi)]
        ni         = self._var_n[f3(r)]

        # block 0: <l_u, n>   block 1: <l_w, n>   block 2: <(w-u)/|w-u|, n>
        self._rows = np.concatenate([
            np.repeat(r, 3), np.repeat(r, 3),                      # l_u, n
            np.repeat(r + E, 3), np.repeat(r + E, 3),              # l_w, n
            np.repeat(r + 2 * E, 3), np.repeat(r + 2 * E, 3),
            np.repeat(r + 2 * E, 3),                               # u, w, n
        ]).astype(np.int32)
        self._cols = np.concatenate([
            li_u, ni,
            li_w, ni,
            vi_u, vi_w, ni,
        ]).astype(np.int32)

    def accept_step(self, X) -> None:
        v = X[self._var_v].reshape(-1, 3)
        self._e_norm = np.linalg.norm(v[self._wi] - v[self._ui], axis=1) + 1e-12

    def _parts(self, X):
        v = X[self._var_v].reshape(-1, 3)
        l = X[self._var_l].reshape(-1, 3)
        n = X[self._var_n].reshape(-1, 3)
        return v, l, n

    def res(self, X) -> np.ndarray:
        v, l, n = self._parts(X)
        E = self._E
        out = np.empty(3 * E)
        out[0:E]       = np.einsum("ij,ij->i", l[self._ui], n)
        out[E:2 * E]   = np.einsum("ij,ij->i", l[self._wi], n)
        out[2 * E:]    = np.einsum("ij,ij->i",
                                   v[self._wi] - v[self._ui], n) / self._e_norm
        return out

    def grad(self, X) -> np.ndarray:
        v, l, n = self._parts(X)
        inv = (1.0 / self._e_norm)[:, None]
        return np.concatenate([
            n.ravel(),                                   # d/dl_u  of <l_u, n>
            l[self._ui].ravel(),                         # d/dn
            n.ravel(),                                   # d/dl_w  of <l_w, n>
            l[self._wi].ravel(),                         # d/dn
            (-n * inv).ravel(),                          # d/du   (normaliser applied)
            (n * inv).ravel(),                           # d/dw
            ((v[self._wi] - v[self._ui]) * inv).ravel(),  # d/dn
        ])
