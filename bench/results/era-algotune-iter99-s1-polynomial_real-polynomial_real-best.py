import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True, parallel=True)
def aberth_roots_numba(coeffs, max_iter=60, tol=1e-12):
    n = len(coeffs) - 1
    an = coeffs[0]

    # Cauchy bound
    max_ratio = 0.0
    for i in range(1, n + 1):
        ratio = abs(coeffs[i] / an)
        if ratio > max_ratio:
            max_ratio = ratio
    bound = 1.0 + max_ratio

    # Initial guesses: real, evenly spaced in [-bound, bound] with small perturbation
    roots = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        x = -bound + 2.0 * bound * i / (n - 1) if n > 1 else 0.0
        roots[i] = x + 0.01 * (1.0j * (i % 5) - 0.003)

    p_vals = np.zeros(n, dtype=np.complex128)
    p_prime_vals = np.zeros(n, dtype=np.complex128)
    corrections = np.zeros(n, dtype=np.complex128)

    for it in range(max_iter):
        # Evaluate polynomial and derivative using Horner's method
        for i in range(n):
            p = coeffs[0]
            dp = 0.0 + 0.0j
            for k in range(1, len(coeffs)):
                dp = dp * roots[i] + p
                p = p * roots[i] + coeffs[k]
            p_vals[i] = p
            p_prime_vals[i] = dp

        max_corr = 0.0
        for i in nb.prange(n):
            s = 0.0 + 0.0j
            ri = roots[i]
            for j in range(n):
                if i != j:
                    diff = ri - roots[j]
                    if abs(diff) < 1e-15:
                        roots[j] += 1e-10 * (1.0 + 1.0j)
                        diff = ri - roots[j]
                    s += 1.0 / diff
            denom = p_prime_vals[i] - p_vals[i] * s
            if abs(denom) < 1e-15:
                denom = 1e-15 + 0.0j
            corr = p_vals[i] / denom
            corrections[i] = corr
            if abs(corr) > max_corr:
                max_corr = abs(corr)

        for i in range(n):
            roots[i] -= corrections[i]

        if max_corr < tol:
            break

    return roots


def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    if coeffs[0] != 1.0:
        coeffs = coeffs / coeffs[0]
    roots = aberth_roots_numba(coeffs)
    roots_real = np.real(roots)
    roots_sorted = np.sort(roots_real)[::-1]
    return roots_sorted.tolist()
