import numpy as np
from numba import njit

@njit(cache=True)
def durand_kerner(coeffs, max_iter=100, tol=1e-14):
    n = len(coeffs) - 1
    c = coeffs.copy()
    # monic
    a0 = c[0]
    if a0 != 1.0:
        c = c / a0

    # Cauchy bound
    max_abs = 0.0
    for i in range(1, n+1):
        if abs(c[i]) > max_abs:
            max_abs = abs(c[i])
    R = 1.0 + max_abs

    # initial guesses: equally spaced on circle radius R, slightly perturbed
    roots = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        theta = 2.0 * np.pi * i / n + 0.01 * (i % 7)
        roots[i] = R * np.exp(1j * theta)

    # Durand-Kerner iteration
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(n):
            # evaluate polynomial at current root
            p = c[0]
            for j in range(1, n+1):
                p = p * roots[i] + c[j]
            # compute denominator: product of (roots[i] - roots[j]) for j != i
            denom = 1.0 + 0.0j
            for j in range(n):
                if j != i:
                    denom *= (roots[i] - roots[j])
            if abs(denom) < 1e-300:
                denom = 1e-300
            delta = p / denom
            roots[i] -= delta
            if abs(delta) > max_change:
                max_change = abs(delta)
        if max_change < tol:
            break

    return roots

@njit(cache=True)
def refine_newton(roots, coeffs, n_iter=5):
    n = len(roots)
    for _ in range(n_iter):
        for i in range(n):
            r = roots[i]
            p = coeffs[0]
            dp = 0.0 + 0.0j
            for j in range(1, n+1):
                dp = dp * r + p
                p = p * r + coeffs[j]
            if abs(dp) > 1e-300:
                roots[i] -= p / dp
    return roots

def solve(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = len(coeffs) - 1
    if n == 0:
        return []

    roots = durand_kerner(coeffs)
    roots = refine_newton(roots, coeffs, 3)

    # all roots are real, take real parts and sort descending
    real_parts = np.real(roots)
    real_parts = np.sort(real_parts)[::-1]
    return real_parts.tolist()
