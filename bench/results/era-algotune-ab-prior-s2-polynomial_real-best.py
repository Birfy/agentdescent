import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def aberth_roots(coeffs, n):
    """Optimized Aberth method for real-rooted polynomials using vectorization."""
    # Normalize
    coeffs = coeffs / coeffs[0]
    
    # Cauchy bound
    R = 1.0 + np.max(np.abs(coeffs[1:]))
    
    # Initial guesses: Chebyshev nodes mapped to [-R, R]
    # Slightly perturbed to avoid symmetry issues
    z = R * 0.95 * np.cos(np.linspace(np.pi, 0, n))
    # Add tiny random perturbation to break symmetry
    for i in range(n):
        z[i] += (i * 1e-10) % 1e-8
    
    # Derivative coefficients
    dp_coeffs = np.empty(n)
    for i in range(n):
        dp_coeffs[i] = coeffs[i] * (n - i)
    
    # Pre-allocate arrays
    p = np.empty(n)
    dp = np.empty(n)
    corr = np.empty(n)
    
    # Iterate
    for _ in range(20):
        # Evaluate polynomial and derivative using vectorized Horner
        # Initialize with leading coefficient
        val = np.full(n, coeffs[0])
        dval = np.zeros(n)
        for j in range(1, n + 1):
            dval = dval * z + val
            val = val * z + coeffs[j]
        p = val
        dp = dval
        
        # Compute corrections using vectorized operations
        max_corr = 0.0
        for i in range(n):
            zi = z[i]
            # Vectorized sum of reciprocals
            diff = zi - z
            # Avoid division by zero
            mask = diff != 0.0
            s = 0.0
            if mask.any():
                s = np.sum(1.0 / diff[mask])
                # Subtract self term if it was included
                if mask[i]:
                    s -= 1.0 / diff[i]
            
            denom = dp[i] - p[i] * s
            if denom != 0.0:
                c = p[i] / denom
                corr[i] = c
                z[i] -= c
                if abs(c) > max_corr:
                    max_corr = abs(c)
            else:
                corr[i] = 0.0
        
        if max_corr < 1e-15:
            break
    
    return z

@njit(cache=True, fastmath=True, parallel=True)
def aberth_roots_parallel(coeffs, n):
    """Parallel version using prange for the correction loop."""
    # Normalize
    coeffs = coeffs / coeffs[0]
    
    # Cauchy bound
    R = 1.0 + np.max(np.abs(coeffs[1:]))
    
    # Initial guesses
    z = R * 0.95 * np.cos(np.linspace(np.pi, 0, n))
    for i in range(n):
        z[i] += (i * 1e-10) % 1e-8
    
    # Derivative coefficients
    dp_coeffs = np.empty(n)
    for i in range(n):
        dp_coeffs[i] = coeffs[i] * (n - i)
    
    # Pre-allocate
    p = np.empty(n)
    dp = np.empty(n)
    corr = np.empty(n)
    
    for _ in range(20):
        # Evaluate polynomial and derivative
        for i in prange(n):
            val = coeffs[0]
            dval = 0.0
            for j in range(1, n + 1):
                dval = dval * z[i] + val
                val = val * z[i] + coeffs[j]
            p[i] = val
            dp[i] = dval
        
        # Compute corrections in parallel
        max_corr = 0.0
        for i in prange(n):
            zi = z[i]
            s = 0.0
            for j in range(n):
                if i != j:
                    diff = zi - z[j]
                    if diff != 0.0:
                        s += 1.0 / diff
            denom = dp[i] - p[i] * s
            if denom != 0.0:
                c = p[i] / denom
                corr[i] = c
                z[i] -= c
                if abs(c) > max_corr:
                    max_corr = abs(c)
            else:
                corr[i] = 0.0
        
        if max_corr < 1e-15:
            break
    
    return z

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    
    # For degree 396, use the optimized Aberth method
    # Try parallel version first, fall back to serial if issues
    try:
        roots = aberth_roots_parallel(coeffs, n)
    except:
        roots = aberth_roots(coeffs, n)
    
    # Clean up: take real parts and sort descending
    roots = np.real(roots)
    roots = np.sort(roots)[::-1]
    return roots.tolist()
