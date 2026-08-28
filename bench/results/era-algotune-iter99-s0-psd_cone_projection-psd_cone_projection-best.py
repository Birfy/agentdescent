import numpy as np
from typing import Any

def solve(problem: dict[str, np.ndarray]) -> dict[str, Any]:
    A = np.asarray(problem['A'], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(A)
    pos = eigvals > 0
    if pos.all():
        X = A
    else:
        vals = eigvals[pos]
        vecs = eigvecs[:, pos]
        X = (vecs * vals) @ vecs.T
    return {'X': X}
