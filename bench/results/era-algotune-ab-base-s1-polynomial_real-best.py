import numpy as np
from numba import njit

@njit(cache=True)
def aberth_roots(coeffs, n):
    # Cauchy bound
    R = 1.0 + np.max(np.abs(coeffs[1:]))
    # Initial roots: equally spaced on circle of radius R
    angles = 2.0 * np.pi * np.arange(n) / n
    z = R * np.exp(1j * angles)

    # Derivative coefficients
    deriv = np.zeros(n, dtype=np.float64)
    for k in range(n):
        deriv[k] = (n - k) * coeffs[k]

    # Preallocate
    p = np.zeros(n, dtype=np.complex128)
    dp = np.zeros(n, dtype=np.complex128)
    ratio = np.zeros(n, dtype=np.complex128)
    z_old = np.zeros(n, dtype=np.complex128)

    for _ in range(50):  # fewer iterations, tighter tolerance
        # Evaluate polynomial and derivative using Horner
        for i in range(n):
            p[i] = coeffs[0]
            dp[i] = deriv[0]
        for c in coeffs[1:]:
            for i in range(n):
                p[i] = p[i] * z[i] + c
        for c in deriv[1:]:
            for i in range(n):
                dp[i] = dp[i] * z[i] + c

        # Ratio p/dp
        max_diff = 0.0
        for i in range(n):
            if abs(dp[i]) < 1e-15:
                dp[i] = 1e-15
            ratio[i] = p[i] / dp[i]

        # Update each root
        for i in range(n):
            s = 0.0j
            zi = z[i]
            for j in range(n):
                if i != j:
                    s += 1.0 / (zi - z[j])
            denom = 1.0 - ratio[i] * s
            if abs(denom) < 1e-15:
                denom = 1e-15
            z[i] = zi - ratio[i] / denom

            diff = abs(z[i] - zi)
            if diff > max_diff:
                max_diff = diff

        if max_diff < 1e-14:
            break

    roots = np.real(z)
    roots.sort()
    return roots[::-1]

def solve(coeffs):
    n = len(coeffs) - 1
    if n == 0:
        return np.array([])

    monic_coeffs = np.asarray(coeffs, dtype=np.float64)
    if monic_coeffs[0] != 1.0:
        monic_coeffs = monic_coeffs / monic_coeffs[0]

    return aberth_roots(monic_coeffs, n)
