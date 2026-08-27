import numpy as np
import numba
from typing import Any

@numba.njit(cache=True, fastmath=True)
def _power_control(G, sigma, P_min, P_max, S_min, max_iter=200, tol=1e-14):
    n = G.shape[0]
    diag_G = np.diag(G).copy()
    off_diag = G.copy()
    for i in range(n):
        off_diag[i, i] = 0.0

    scale = S_min / diag_G
    P = P_min.copy()

    for _ in range(max_iter):
        interference = off_diag @ P
        required = scale * (sigma + interference)
        new_P = np.clip(required, P_min, P_max)
        if np.abs(new_P - P).max() < tol:
            P = new_P
            break
        P = new_P

    return P

def solve(problem: dict[str, Any]) -> dict[str, Any]:
    G = np.asarray(problem['G'], dtype=np.float64)
    sigma = np.asarray(problem['σ'], dtype=np.float64)
    P_min = np.asarray(problem['P_min'], dtype=np.float64)
    P_max = np.asarray(problem['P_max'], dtype=np.float64)
    S_min = float(problem['S_min'])

    P = _power_control(G, sigma, P_min, P_max, S_min)

    return {'P': P.tolist(), 'objective': float(np.sum(P))}
