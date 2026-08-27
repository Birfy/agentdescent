import numpy as np

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    if n == 0:
        return []
    if n == 1:
        return [-coeffs[1]]
    
    roots = np.roots(coeffs)
    # All roots are real, so just take real parts (imaginary parts are negligible)
    real_roots = np.real(roots)
    real_roots.sort()
    return real_roots[::-1].tolist()
