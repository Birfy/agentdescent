import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True, parallel=True)
def aberth_roots_parallel_sorted(coeff, z, max_iter=100, tol=1e-12, eps=1e-14):
    """Parallel Aberth method with early convergence and in-place sort."""
    n = len(z)
    p = np.empty(n, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)
    S = np.empty(n, dtype=np.float64)
    dz_arr = np.empty(n, dtype=np.float64)
    
    for _ in range(max_iter):
        # Horner evaluation for all points simultaneously (parallel)
        for j in prange(n):
            p[j] = coeff[0]
            dp[j] = 0.0
            for i in range(1, len(coeff)):
                dp[j] = dp[j] * z[j] + p[j]
                p[j] = p[j] * z[j] + coeff[i]
        
        # Compute sum of reciprocals of differences (parallel)
        for j in prange(n):
            s = 0.0
            zj = z[j]
            for k in range(n):
                if k != j:
                    diff = zj - z[k]
                    if abs(diff) < eps:
                        diff = eps if diff >= 0 else -eps
                    s += 1.0 / diff
            S[j] = s
        
        # Aberth update (parallel)
        for j in prange(n):
            if abs(dp[j]) < eps:
                dz = 0.0
            else:
                ratio = p[j] / dp[j]
                denom = 1.0 - ratio * S[j]
                if abs(denom) < eps:
                    dz = ratio
                else:
                    dz = ratio / denom
            dz_arr[j] = dz
            z[j] -= dz
        
        # Check convergence
        max_dz = 0.0
        for j in range(n):
            d = abs(dz_arr[j])
            if d > max_dz:
                max_dz = d
        
        if max_dz < tol:
            break
    
    # Sort descending - insertion sort (fast for nearly sorted data)
    for i in range(1, n):
        key = z[i]
        j = i - 1
        while j >= 0 and z[j] < key:
            z[j + 1] = z[j]
            j -= 1
        z[j + 1] = key
    return z

def solve(problem):
    coeff = np.asarray(problem, dtype=np.float64)
    n = len(coeff) - 1
    if n == 0:
        return []
    
    # Handle leading zero coefficients
    if coeff[0] == 0:
        idx = 0
        while idx < len(coeff) and coeff[idx] == 0:
            idx += 1
        if idx == len(coeff):
            return []
        coeff = coeff[idx:]
        n = len(coeff) - 1
        if n == 0:
            return []
    
    # Root bound using Cauchy's bound
    R = 1.0 + np.max(np.abs(coeff[1:])) / np.abs(coeff[0])
    
    # Chebyshev nodes on [-R, R] - excellent initial guesses for real-rooted polynomials
    k = np.arange(1, n + 1)
    z = R * np.cos(np.pi * (2 * k - 1) / (2 * n))
    
    # Run parallel Aberth with sorted output
    roots = aberth_roots_parallel_sorted(coeff, z)
    
    return roots.tolist()

PROMISE: 10
