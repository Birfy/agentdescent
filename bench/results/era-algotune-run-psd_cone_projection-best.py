import numpy as np
from scipy.linalg import eigh as scipy_eigh

def solve(problem):
    A = np.ascontiguousarray(problem['A'], dtype=np.float64)
    w, V = scipy_eigh(A, driver='evd', overwrite_a=True)
    np.maximum(w, 0.0, out=w)
    X = (V * w) @ V.T
    return {'X': X}
