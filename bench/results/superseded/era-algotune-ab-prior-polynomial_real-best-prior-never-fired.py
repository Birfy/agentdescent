import numpy as np

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    if n == 0:
        return []
    if n == 1:
        return [-coeffs[1] / coeffs[0]]

    # Normalize to monic
    a = coeffs / coeffs[0]

    # Root bound using Cauchy's bound
    R = 1.0 + np.max(np.abs(a[1:]))

    # Use scaled Chebyshev nodes for better initial distribution
    k = np.arange(n, dtype=np.float64)
    t = R * np.cos(np.pi * (2 * k + 1) / (2 * n))
    # Add tiny deterministic perturbation to avoid symmetry issues
    t += 1e-6 * np.sin(k * 7.3 + 1.7)
    z = t.astype(np.float64)

    max_iter = 15
    tol = 1e-12

    # Pre-allocate arrays
    p = np.empty(n, dtype=np.float64)
    p_prime = np.empty(n, dtype=np.float64)
    diff = np.empty((n, n), dtype=np.float64)

    for _ in range(max_iter):
        # Horner evaluation for P(z) and P'(z) - vectorized
        p.fill(a[0])
        p_prime.fill(0.0)
        for c in a[1:]:
            np.multiply(p_prime, z, out=p_prime)
            p_prime += p
            np.multiply(p, z, out=p)
            p += c

        # Compute sum_{j!=i} 1/(z_i - z_j) using vectorized operations
        # Use broadcasting but with memory-efficient approach
        diff = z[:, None] - z[None, :]
        np.fill_diagonal(diff, 1.0)
        np.reciprocal(diff, out=diff)
        np.fill_diagonal(diff, 0.0)
        s = diff.sum(axis=1)

        # Aberth update
        denom = p_prime - p * s
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = np.where(denom != 0, p / denom, 0.0)
        z -= delta

        if np.max(np.abs(delta)) < tol:
            break

    # Real parts, sort descending
    roots = np.sort(np.real(z))[::-1]
    return roots.tolist()
