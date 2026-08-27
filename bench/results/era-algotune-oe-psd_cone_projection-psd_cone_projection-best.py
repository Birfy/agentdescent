import numpy as np
from typing import Any

def solve(problem: dict[str, np.ndarray]) -> dict[str, Any]:
    A = np.asarray(problem['A'])
    eigvals, eigvecs = np.linalg.eigh(A)
    # Only keep positive eigenvalues
    pos = eigvals > 0
    if not np.any(pos):
        return {'X': np.zeros_like(A)}
    eigvals_pos = eigvals[pos]
    eigvecs_pos = eigvecs[:, pos]
    X = (eigvecs_pos * eigvals_pos) @ eigvecs_pos.T
    return {'X': X}
