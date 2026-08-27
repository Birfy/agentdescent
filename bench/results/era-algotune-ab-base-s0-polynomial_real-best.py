import numpy as np
from scipy.linalg import eigvals

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    # Build companion matrix
    if coeffs[0] != 1.0:
        coeffs = coeffs / coeffs[0]
    # Companion matrix: first row is -coeffs[1:]/coeffs[0], subdiagonal ones
    A = np.zeros((n, n))
    if n > 1:
        A[0, :] = -coeffs[1:] / coeffs[0]
        A[1:, :-1] = np.eye(n-1)
    else:
        A[0,0] = -coeffs[1]/coeffs[0]
    roots = eigvals(A, check_finite=False)
    # Since all roots are real, take real part
    roots = np.real(roots)
    roots = np.sort(roots)[::-1]
    return roots.tolist()
