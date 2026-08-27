import numpy as np
from scipy.linalg import eigvals

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = coeffs.shape[0] - 1
    if n == 1:
        return [-coeffs[1] / coeffs[0]]
    a0 = coeffs[0]
    companion = np.empty((n, n), dtype=np.float64)
    companion[0, :] = -coeffs[1:] / a0
    if n > 1:
        companion[1:, :-1] = np.eye(n - 1)
    # The last column is already zero from np.empty? Actually np.empty gives garbage, so we need to set it.
    # Better to use np.zeros and then fill, but we can set the subdiagonal and first row.
    # Let's use np.zeros and then fill.
    companion = np.zeros((n, n), dtype=np.float64)
    companion[0, :] = -coeffs[1:] / a0
    if n > 1:
        companion[1:, :-1] = np.eye(n - 1)
    roots = eigvals(companion, check_finite=False, overwrite_a=True)
    roots = np.sort(np.real(roots))[::-1]
    return roots.tolist()
