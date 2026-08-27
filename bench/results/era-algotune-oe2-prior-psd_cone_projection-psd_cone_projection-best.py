import numpy as np
from typing import Any

def solve(problem: dict[str, np.ndarray]) -> dict[str, Any]:
    A = np.asarray(problem['A'])
    
    # Use the LAPACK divide-and-conquer eigensolver via UPLO='L'
    # This is the fastest for n=349 symmetric matrices
    eigvals, eigvecs = np.linalg.eigh(A, UPLO='L')
    
    # Clip eigenvalues to non-negative (in-place)
    np.maximum(eigvals, 0, out=eigvals)
    
    # Reconstruct X = V * diag(max(lambda,0)) * V^T
    # Using BLAS-3 gemm for optimal performance
    X = (eigvecs * eigvals) @ eigvecs.T
    
    return {'X': X}
