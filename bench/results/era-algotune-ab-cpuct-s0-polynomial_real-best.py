import numpy as np
from scipy.linalg import eigvals

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    
    # Normalize if needed
    if coeffs[0] != 1.0:
        coeffs = coeffs / coeffs[0]
    
    # Build companion matrix directly
    A = np.zeros((n, n))
    if n > 1:
        A[1:, :-1] = np.eye(n-1)
    A[0, :] = -coeffs[1:]
    
    # Use eigvals with check_finite=False for speed
    roots = eigvals(A, check_finite=False)
    roots = np.real(roots)
    roots = np.sort(roots)[::-1]
    return roots.tolist()
