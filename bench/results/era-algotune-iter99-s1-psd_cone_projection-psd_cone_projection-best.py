import numpy as np
from typing import Any

def solve(problem: dict[str, np.ndarray]) -> dict[str, Any]:
    A = problem['A']
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    
    # Use eigh with LAPACK syevd driver - optimal for symmetric matrices
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # Clip negative eigenvalues to zero in-place
    np.maximum(eigvals, 0, out=eigvals)
    
    # Matrix multiplication: (V * λ) @ V^T
    # This avoids creating an intermediate full diagonal matrix
    X = (eigvecs * eigvals) @ eigvecs.T
    
    return {'X': X}
