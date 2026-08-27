import numpy as np
from numba import njit

@njit(cache=True)
def aberth_step(coeffs, z):
    n = z.shape[0]
    p = np.zeros(n, dtype=np.complex128)
    dp = np.zeros(n, dtype=np.complex128)
    
    # Horner evaluation
    for i in range(n):
        p_val = coeffs[0]
        dp_val = 0.0 + 0.0j
        for j in range(1, coeffs.shape[0]):
            dp_val = dp_val * z[i] + p_val
            p_val = p_val * z[i] + coeffs[j]
        p[i] = p_val
        dp[i] = dp_val
    
    # Sum of 1/(z_i - z_j)
    s = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = z[i] - z[j]
                if np.abs(diff) > 1e-60:
                    s[i] += 1.0 / diff
    
    # Update
    max_delta = 0.0
    for i in range(n):
        denom = dp[i] - p[i] * s[i]
        if np.abs(denom) > 1e-60:
            delta = p[i] / denom
        else:
            delta = p[i] / (1e-30 + 0.0j)
        z[i] -= delta
        max_delta = max(max_delta, np.abs(delta))
    return z, max_delta

@njit(cache=True)
def solve_numba(coeffs_float):
    n = len(coeffs_float) - 1
    if n == 0:
        return np.empty(0, dtype=np.float64)
    
    # Normalize leading coefficient to 1
    coeffs = np.zeros(n + 1, dtype=np.complex128)
    lead = coeffs_float[0]
    for i in range(n + 1):
        coeffs[i] = coeffs_float[i] / lead
    
    # Cauchy bound
    R = 1.0
    for i in range(1, n + 1):
        R = max(R, np.abs(coeffs[i]))
    R += 1.0
    
    # Initial guess: roots of unity on circle radius R
    z = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        z[i] = R * np.cos(angle) + 1e-6j * np.sin(angle)
    
    # Iterate
    for _ in range(100):
        z, max_delta = aberth_step(coeffs, z)
        if max_delta < 1e-14:
            break
    
    # Extract real parts (roots are real)
    roots = np.real(z)
    # Sort descending
    roots = np.sort(roots)[::-1]
    return roots

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    roots = solve_numba(coeffs)
    return roots.tolist()
