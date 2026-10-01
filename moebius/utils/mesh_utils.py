"""
Mesh and line-congruence utilities needed by the torsal energies.
Ported from the old hanan/geometry/utils.py — only the functions not present
in the new hanan package are kept here.
"""

import numpy as np
from scipy.optimize import minimize


# ── basic vector utilities ───────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / np.linalg.norm(v)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def vec_dot(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    if v1.ndim == 1 and v2.ndim == 1:
        return v1 @ v2
    return np.einsum("ij,ij->i", v1, v2)


def flat_array_variables(arr: np.ndarray, n: int = 3) -> np.ndarray:
    """Return interleaved 3× indices: [3a, 3a+1, 3a+2, 3b, 3b+1, ...]."""
    stacked = np.stack([arr * n + k for k in range(n)], axis=1)
    return stacked.flatten()


def planarity_check(t1: np.ndarray, lt1: np.ndarray, lc: np.ndarray) -> np.ndarray:
    """Planarity measure: |[t1, lt1, lc]| (triple product norm)."""
    return np.abs(vec_dot(np.cross(t1, lt1), lc))


# ── torsal utilities ─────────────────────────────────────────────────────────

def lc_info_at_grid_points(l: np.ndarray):
    """Compute lc (barycenter), lu and lv for all quad faces from an (m,n,3) grid.

    Returns
    -------
    lc  : (F, 3) – line congruence at each face centre
    lu  : (F, 3) – finite-difference u derivative of l at each face
    lv  : (F, 3) – finite-difference v derivative of l at each face
    """
    l0 = l[:-1, :-1]
    l1 = l[:-1, 1:]
    l2 = l[1:,  1:]
    l3 = l[1:,  :-1]
    lc = (l0 + l1 + l2 + l3) / 4
    lu = l2 - l0
    lv = l1 - l3
    return lc, lu, lv


def _approximate_torsal(lc, lu, lv, du, dv):
    """Fallback for negative discriminant: find optimal (u, v) torsal pair."""
    F = len(lc)
    opt_t1 = np.zeros((F, 2))
    opt_t2 = np.zeros((F, 2))

    def energy(x):
        u, v = x
        t = u * du_i + v * dv_i
        lt = u * lu_i + v * lv_i
        return vec_dot(t, np.cross(lt, lc_i))**2

    for i in range(F):
        lc_i, lu_i, lv_i, du_i, dv_i = lc[i], lu[i], lv[i], du[i], dv[i]
        res1 = minimize(energy, [1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-6})
        res2 = minimize(energy, [0.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-6})
        opt_t1[i] = res1.x / (np.linalg.norm(res1.x) + 1e-12)
        opt_t2[i] = res2.x / (np.linalg.norm(res2.x) + 1e-12)

    return opt_t1, opt_t2


def torsal_directions(lc, lu, lv, du, dv):
    """Compute torsal directions (t1, t2) and their parameter components.

    Solves  ut² [du, lu, lc] + ut·vt ([du,lv,lc]+[dv,lu,lc]) + vt² [dv,lv,lc] = 0.

    Returns
    -------
    t1, t2  : (F,3) unit torsal directions
    ut1, vt1, ut2, vt2 : parameter-space components (normalised so ||t||=1)
    neg_idx : indices where discriminant was ≤ 0 (approximate fallback used)
    """
    lc = unit(lc)

    g0 = vec_dot(du, np.cross(lu, lc))
    g1 = vec_dot(du, np.cross(lv, lc)) + vec_dot(dv, np.cross(lu, lc))
    g2 = vec_dot(dv, np.cross(lv, lc))

    disc = g1**2 - 4 * g0 * g2

    ut1 = np.zeros(len(lc))
    vt1 = np.zeros(len(lc))
    ut2 = np.zeros(len(lc))
    vt2 = np.zeros(len(lc))

    pos = disc > 1e-8
    ut1[pos] = -g1[pos] + np.sqrt(disc[pos])
    ut2[pos] = -g1[pos] - np.sqrt(disc[pos])
    vt1[pos] = vt2[pos] = 2 * g0[pos]

    neg_idx = np.where(~pos)[0]
    if neg_idx.size > 0:
        fallback = _approximate_torsal(lc[neg_idx], lu[neg_idx], lv[neg_idx],
                                       du[neg_idx], dv[neg_idx])
        ut1[neg_idx] = fallback[0][:, 0]
        vt1[neg_idx] = fallback[0][:, 1]
        ut2[neg_idx] = fallback[1][:, 0]
        vt2[neg_idx] = fallback[1][:, 1]

    t1 = ut1[:, None] * du + vt1[:, None] * dv
    t2 = ut2[:, None] * du + vt2[:, None] * dv

    nrm1 = np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    nrm2 = np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12

    ut1 /= nrm1[:, 0]; vt1 /= nrm1[:, 0]
    ut2 /= nrm2[:, 0]; vt2 /= nrm2[:, 0]
    t1  /= nrm1; t2  /= nrm2

    return t1, t2, ut1, vt1, ut2, vt2, neg_idx
