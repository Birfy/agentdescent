import numpy as np
from numba import njit

@njit(cache=True)
def aberth_roots(coeffs, max_iter=200, tol=1e-14):
    n = len(coeffs) - 1
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([-coeffs[1]/coeffs[0]])
    if n == 2:
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        disc = b*b - 4*a*c
        if disc < 0:
            return np.empty(0, dtype=np.float64)
        sqrt_disc = np.sqrt(disc)
        r1 = (-b + sqrt_disc)/(2*a)
        r2 = (-b - sqrt_disc)/(2*a)
        return np.array([max(r1,r2), min(r1,r2)])
    
    # Cauchy bound for radius
    max_ratio = 0.0
    for i in range(1, n+1):
        max_ratio = max(max_ratio, abs(coeffs[i]/coeffs[0]))
    R = max(1.0, max_ratio + 1.0)
    
    # Initialize on circle with slight offset to avoid symmetry
    roots = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        angle = 2*np.pi * i / n + 0.1/n
        roots[i] = R * np.exp(1j*angle)
    
    # Precompute coefficients
    c = coeffs.copy()
    
    # Aberth–Ehrlich iteration with vectorized updates for speed
    for _ in range(max_iter):
        max_delta = 0.0
        # Evaluate P and P' at each root using Horner
        for i in range(n):
            zi = roots[i]
            p = c[0]
            dp = 0.0j
            for k in range(1, n+1):
                dp = dp * zi + p
                p = p * zi + c[k]
            
            # Compute sum of 1/(zi-roots[j]) for j != i
            s = 0.0j
            for j in range(n):
                if j != i:
                    diff = zi - roots[j]
                    s += 1.0 / diff
            
            denom = dp - p * s
            if abs(denom) < 1e-300:
                continue
            delta = p / denom
            roots[i] -= delta
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        if max_delta < tol:
            break
    
    # Take real parts and sort descending
    real_roots = np.sort(roots.real)[::-1]
    return real_roots

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    roots = aberth_roots(coeffs)
    return roots.tolist()
