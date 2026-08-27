import numpy as np
import scipy.linalg

def solve(problem):
    coefficients = np.asarray(problem, dtype=np.float64)
    n = coefficients.size - 1
    if n == 0:
        return []
    if n == 1:
        return [(-coefficients[1] / coefficients[0]).item()]
    leading = coefficients[0]
    if leading == 0:
        idx = np.nonzero(coefficients)[0]
        if idx.size == 0:
            return []
        coefficients = coefficients[idx[0]:]
        return solve(coefficients)
    coeffs = coefficients / leading
    comp = np.zeros((n, n), dtype=np.float64)
    comp[0, :] = -coeffs[1:]
    if n > 1:
        comp[1:, :-1] = np.eye(n-1)
    roots = scipy.linalg.eigvals(comp, check_finite=False, overwrite_a=True)
    roots = np.real(roots)
    roots = np.sort(roots)[::-1]
    return roots.tolist()
