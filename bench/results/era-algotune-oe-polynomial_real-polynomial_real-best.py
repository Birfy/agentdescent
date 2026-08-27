import numpy as np

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    # numpy's roots returns roots in descending order already
    roots = np.roots(coeffs)
    # Take real part (all roots are real by problem statement)
    roots = np.real(roots)
    # Sort in descending order
    roots.sort()
    return roots[::-1].tolist()
