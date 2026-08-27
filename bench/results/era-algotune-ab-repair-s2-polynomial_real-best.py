import numpy as np
import numba

@numba.njit(cache=True, fastmath=True, parallel=True)
def aberth_step_vectorized(coeffs, roots):
    n = roots.shape[0]
    new_roots = np.empty_like(roots)
    for i in numba.prange(n):
        x = roots[i]
        # Evaluate p and p' using Horner
        p = coeffs[0]
        dp = 0.0
        for j in range(1, len(coeffs)):
            dp = dp * x + p
            p = p * x + coeffs[j]
        if dp == 0.0:
            new_roots[i] = x
            continue
        corr = p / dp
        sum_inv = 0.0
        for j in range(n):
            if j != i:
                diff = x - roots[j]
                if diff != 0.0:
                    sum_inv += 1.0 / diff
        denom = 1.0 - corr * sum_inv
        if denom == 0.0:
            new_roots[i] = x
        else:
            new_roots[i] = x - corr / denom
    return new_roots

@numba.njit(cache=True, fastmath=True, parallel=True)
def newton_polish(coeffs, roots, iterations):
    n = roots.shape[0]
    for _ in range(iterations):
        for i in numba.prange(n):
            x = roots[i]
            p = coeffs[0]
            dp = 0.0
            for j in range(1, len(coeffs)):
                dp = dp * x + p
                p = p * x + coeffs[j]
            if dp != 0.0:
                roots[i] = x - p / dp

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1

    if n == 1:
        return [(-coeffs[1] / coeffs[0]).item()]

    # Normalize to monic
    a = coeffs / coeffs[0]

    # Cauchy root bound
    R = 1.0 + np.max(np.abs(a[1:]))

    # Initial roots: Chebyshev nodes scaled to [-R, R]
    k = np.arange(n) + 0.5
    roots = R * np.cos(np.pi * k / n)

    # Aberth iterations
    max_iter = 30
    tol = 1e-14
    for _ in range(max_iter):
        new_roots = aberth_step_vectorized(a, roots)
        delta = np.max(np.abs(new_roots - roots))
        roots = new_roots
        if delta < tol:
            break

    # Sort descending
    order = np.argsort(roots)[::-1]
    roots = roots[order]

    # Newton polish
    newton_polish(a, roots, 5)

    return roots.tolist()
