import numpy as np
from typing import Any

def solve(problem: dict[str, np.ndarray]) -> dict[str, Any]:
    A = np.asarray(problem['A'])
    # Use eigh (symmetric) for speed and stability
    eigvals, eigvecs = np.linalg.eigh(A)
    # Clip negative eigenvalues to zero in-place to reduce allocations
    np.maximum(eigvals, 0, out=eigvals)
    # Reconstruct X = V * diag(max(λ,0)) * V^T
    X = (eigvecs * eigvals) @ eigvecs.T
    return {'X': X}
