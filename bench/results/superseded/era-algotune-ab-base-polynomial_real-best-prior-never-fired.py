import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def _aberth_roots(coeffs, guesses, max_iter=50, tol=1e-14):
    n = len(guesses)
    for _ in range(max_iter):
        p_vals = np.zeros(n, dtype=np.complex128)
        p_prime = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            p = coeffs[0] + 0.0j
            dp = 0.0 + 0.0j
            for c in coeffs[1:]:
                dp = dp * guesses[i] + p
                p = p * guesses[i] + c
            p_vals[i] = p
            p_prime[i] = dp

        sum_inv = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            s = 0.0 + 0.0j
            xi = guesses[i]
            for j in range(n):
                if i != j:
                    s += 1.0 / (xi - guesses[j])
            sum_inv[i] = s

        max_corr = 0.0
        for i in range(n):
            denom = p_prime[i] - p_vals[i] * sum_inv[i]
            if abs(denom) < 1e-300:
                continue
            corr = p_vals[i] / denom
            guesses[i] -= corr
            if abs(corr) > max_corr:
                max_corr = abs(corr)

        if max_corr < tol:
            break

    return guesses

@numba.jit(nopython=True, cache=True)
def _solve_numba(coeffs):
    n = len(coeffs) - 1
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([-coeffs[1] / coeffs[0]], dtype=np.float64)

    coeffs_norm = coeffs / coeffs[0]

    # Use a tighter root bound based on Cauchy
    R = 1.0 + np.max(np.abs(coeffs_norm[1:]))

    # Deterministic initial guesses on a circle
    # Use explicit loop to avoid linspace issues
    guesses = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        guesses[i] = R * (np.cos(angle) + 1j * np.sin(angle))

    # Slight deterministic perturbation to avoid symmetry issues
    # Use a simple deterministic pattern instead of random
    for i in range(n):
        guesses[i] += 1e-8 * (i + 1) * (1.0 + 1.0j)

    roots = _aberth_roots(coeffs_norm, guesses)

    # Real roots expected, discard tiny imaginary parts
    roots = np.real(roots)
    roots = np.sort(roots)[::-1]
    return roots

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    roots = _solve_numba(coeffs)
    return roots.tolist()
