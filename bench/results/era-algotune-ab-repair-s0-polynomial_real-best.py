import numpy as np
from numba import jit

@jit(nopython=True)
def durand_kerner(coeffs, max_iter=100, tol=1e-12):
    n = len(coeffs) - 1
    # initial guesses on a circle
    roots = np.exp(2j * np.pi * np.arange(1, n+1) / n) * (1 + 0.4j)
    for _ in range(max_iter):
        new_roots = roots.copy()
        for i in range(n):
            # manual polynomial evaluation (Horner's method)
            num = coeffs[0]
            for k in range(1, len(coeffs)):
                num = num * roots[i] + coeffs[k]
            den = 1.0
            for j in range(n):
                if i != j:
                    den *= (roots[i] - roots[j])
            new_roots[i] = roots[i] - num / den
        if np.max(np.abs(new_roots - roots)) < tol:
            break
        roots = new_roots
    return roots

def solve(coeffs):
    """Solve polynomial with given coefficients (highest degree first)."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    return durand_kerner(coeffs)
