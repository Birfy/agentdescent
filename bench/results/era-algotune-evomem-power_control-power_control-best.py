import numpy as np
from typing import Any

def solve(problem: dict[str, Any]) -> dict[str, Any]:
    G = np.asarray(problem["G"], dtype=float)
    sigma = np.asarray(problem["σ"], dtype=float)
    P_min = np.asarray(problem["P_min"], dtype=float)
    P_max = np.asarray(problem["P_max"], dtype=float)
    S_min = float(problem["S_min"])

    n = G.shape[0]
    diag_G = np.diag(G)
    ratio = S_min / diag_G

    # Build the linear system: (I - ratio * G + diag(ratio*diag_G)) P = ratio * sigma
    A = -ratio[:, None] * G
    A.flat[:: n + 1] += 1.0 + ratio

    b = ratio * sigma

    P = np.linalg.solve(A, b)
    P = np.clip(P, P_min, P_max)

    return {"P": P.tolist(), "objective": float(np.sum(P))}
