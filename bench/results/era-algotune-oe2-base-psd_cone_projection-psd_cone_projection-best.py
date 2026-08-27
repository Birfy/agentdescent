import numpy as np
from typing import Any

def solve(problem: dict[str, np.ndarray]) -> dict[str, Any]:
    A = np.asarray(problem['A'])
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)
    X = (eigvecs * eigvals) @ eigvecs.T
    return {'X': X}
