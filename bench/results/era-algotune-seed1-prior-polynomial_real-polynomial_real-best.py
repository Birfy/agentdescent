import numpy as np
from scipy.linalg import eigvals

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    if n == 0:
        return []
    
    # Normalize to monic polynomial
    coeffs = coeffs / coeffs[-1]
    
    # Build companion matrix using the standard form (subdiagonal ones)
    # This is faster to construct and scipy's eigvals is highly optimized
    companion = np.zeros((n, n), dtype=np.float64)
    if n > 1:
        # Use numpy's diagonal assignment for speed
        companion[np.arange(1, n), np.arange(0, n-1)] = 1.0
        companion[:, -1] = -coeffs[:-1][::-1]
    else:
        companion[0, 0] = -coeffs[0]
    
    # Compute eigenvalues with scipy's LAPACK wrapper (faster than np.linalg)
    roots = eigvals(companion)
    roots = np.real(roots)
    
    # Newton refinement - 2 iterations with vectorized Horner
    # Use descending coefficients (leading first) for Horner evaluation
    desc = coeffs[::-1]
    for _ in range(2):
        p = np.zeros_like(roots)
        dp = np.zeros_like(roots)
        for c in desc:
            dp = dp * roots + p
            p = p * roots + c
        mask = np.abs(dp) > 1e-12
        roots[mask] = roots[mask] - p[mask] / dp[mask]
    
    # Sort descending
    roots = np.sort(roots)[::-1]
    return roots.tolist()
