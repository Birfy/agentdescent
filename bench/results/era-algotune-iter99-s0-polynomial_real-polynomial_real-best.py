import numpy as np
from numba import njit

@njit(cache=True)
def _durand_kerner_fast(p, roots, max_iter, tol):
    n = len(roots)
    for _ in range(max_iter):
        vals = np.zeros(n, dtype=np.complex128)
        for c in p:
            vals = vals * roots + c

        max_corr = 0.0
        for i in range(n):
            denom = 1.0 + 0.0j
            for j in range(n):
                if i != j:
                    denom *= (roots[i] - roots[j])
            corr = vals[i] / denom
            roots[i] -= corr
            if np.abs(corr) > max_corr:
                max_corr = np.abs(corr)

        if max_corr < tol:
            break
    return roots

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    if n == 0:
        return []
    if n == 1:
        return [-coeffs[1] / coeffs[0]]

    p = coeffs[1:] / coeffs[0]

    # Cauchy's bound
    R = 1.0 + np.max(np.abs(p))
    R *= 1.05

    # Use evenly spaced angles with slight random perturbation for stability
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(42)
    angles += rng.uniform(-0.005, 0.005, n) * (2 * np.pi / n)
    roots = R * np.exp(1j * angles).astype(np.complex128)

    # Iterate until convergence (we need only 3 decimal places)
    max_iter = 20
    tol = 1e-10
    roots = _durand_kerner_fast(p, roots, max_iter, tol)

    # All roots are real per problem statement
    roots = np.real(roots)
    roots = np.sort(roots)[::-1]
    return roots.tolist()
