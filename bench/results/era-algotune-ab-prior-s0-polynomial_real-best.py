import numpy as np
from numba import jit, prange

@jit(nopython=True, cache=True, parallel=True)
def durand_kerner_parallel(coeffs, max_iter=100, tol=1e-10):
    n = len(coeffs) - 1
    if n == 0:
        return np.empty(0)
    if n == 1:
        return np.array([-coeffs[1]/coeffs[0]])
    
    # Normalize leading coefficient
    a0 = coeffs[0]
    c = coeffs[1:] / a0
    
    # Initial guesses: on a circle of radius R (Aberth-style)
    R = 1.0 + np.max(np.abs(c))
    roots = R * np.exp(2j * np.pi * np.arange(n) / n)
    
    for _ in range(max_iter):
        # Evaluate polynomial at all roots simultaneously using Horner's method
        vals = np.empty(n, dtype=np.complex128)
        for i in prange(n):
            val = c[0]
            for k in range(1, n):
                val = val * roots[i] + c[k]
            vals[i] = val * roots[i] + 1.0
        
        # Compute denominators (Weierstrass correction)
        new_roots = roots.copy()
        for i in range(n):
            denom = 1.0
            for j in range(n):
                if j != i:
                    denom *= (roots[i] - roots[j])
            if denom != 0:
                new_roots[i] = roots[i] - vals[i] / denom
        
        # Check convergence
        if np.max(np.abs(new_roots - roots)) < tol:
            roots = new_roots
            break
        roots = new_roots
    
    return np.real(roots)

@jit(nopython=True, cache=True)
def durand_kerner_sequential(coeffs, max_iter=100, tol=1e-10):
    n = len(coeffs) - 1
    if n == 0:
        return np.empty(0)
    if n == 1:
        return np.array([-coeffs[1]/coeffs[0]])
    
    # Normalize leading coefficient
    a0 = coeffs[0]
    c = coeffs[1:] / a0
    
    # Initial guesses: on a circle of radius R
    R = 1.0 + np.max(np.abs(c))
    roots = R * np.exp(2j * np.pi * np.arange(n) / n)
    
    for _ in range(max_iter):
        new_roots = roots.copy()
        for i in range(n):
            # evaluate polynomial at roots[i]
            val = c[0]
            for k in range(1, n):
                val = val * roots[i] + c[k]
            val = val * roots[i] + 1.0
            
            # denominator
            denom = 1.0
            for j in range(n):
                if j != i:
                    denom *= (roots[i] - roots[j])
            if denom != 0:
                new_roots[i] = roots[i] - val / denom
        
        # check convergence
        if np.max(np.abs(new_roots - roots)) < tol:
            roots = new_roots
            break
        roots = new_roots
    
    return np.real(roots)

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    
    # Trim leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0.0:
        coeffs = coeffs[1:]
    
    n = len(coeffs) - 1
    if n == 0:
        return []
    if n == 1:
        return [-coeffs[1]/coeffs[0]]
    
    # Use parallel version for large n, sequential for small
    if n > 50:
        roots = durand_kerner_parallel(coeffs)
    else:
        roots = durand_kerner_sequential(coeffs)
    
    # Sort in descending order
    roots = np.sort(roots)[::-1]
    return roots.tolist()
