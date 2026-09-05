import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _solve_numba(coeffs, n):
    """Numba-accelerated root finding for real-rooted polynomials using Durand-Kerner method."""
    if n == 0:
        return np.empty(0)
    if n == 1:
        return np.array([-coeffs[1] / coeffs[0]])
    if n == 2:
        a, b, c = coeffs
        disc = b*b - 4*a*c
        if disc < 0:
            return np.empty(0)
        sqrt_disc = np.sqrt(disc)
        r1 = (-b + sqrt_disc) / (2*a)
        r2 = (-b - sqrt_disc) / (2*a)
        if r1 >= r2:
            return np.array([r1, r2])
        else:
            return np.array([r2, r1])

    # Normalize to monic
    p = coeffs[1:] / coeffs[0]
    
    # Cauchy bound
    R = 1.0 + np.max(np.abs(p))
    
    # Initial guesses: equally spaced on circle of radius R
    # Use complex roots for Durand-Kerner, but for real-rooted polynomials
    # we can use real initial guesses spread across [-R, R]
    # Better: use Chebyshev-Lobatto points for better convergence
    k = np.arange(n, dtype=np.float64)
    x = R * np.cos(np.pi * (2*k + 1) / (2*n))
    
    max_iter = 50
    tol = 1e-14
    
    # Durand-Kerner iteration
    # For real-rooted polynomials, we can keep everything real
    # The iteration: x_i_new = x_i - f(x_i) / prod_{j!=i}(x_i - x_j)
    
    # Pre-allocate arrays
    f = np.empty(n)
    prod = np.empty(n)
    
    for _ in range(max_iter):
        # Evaluate polynomial at all points using Horner
        for i in range(n):
            val = 1.0
            for c in p:
                val = val * x[i] + c
            f[i] = val
        
        # Compute corrections
        max_corr = 0.0
        for i in range(n):
            prod_val = 1.0
            xi = x[i]
            for j in range(n):
                if i != j:
                    prod_val *= (xi - x[j])
            
            if abs(prod_val) > 1e-300:
                corr = f[i] / prod_val
            else:
                corr = 0.0
            
            x[i] = xi - corr
            if abs(corr) > max_corr:
                max_corr = abs(corr)
        
        if max_corr < tol:
            break
    
    # Sort descending
    x_sorted = np.sort(x)[::-1]
    return x_sorted

def solve(problem):
    """Approximate all real roots of a polynomial with real roots."""
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    
    if n == 0:
        return []
    
    roots = _solve_numba(coeffs, n)
    return roots.tolist()
