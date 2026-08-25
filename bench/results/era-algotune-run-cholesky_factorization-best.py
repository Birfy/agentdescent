import numpy as np
from scipy.linalg.lapack import dpotrf

def solve(problem):
    A = np.array(problem['matrix'], dtype=np.float64, copy=True, order='F')
    L, info = dpotrf(A, lower=1, overwrite_a=True, clean=1)
    if info != 0:
        raise np.linalg.LinAlgError("Cholesky factorization failed.")
    return {'Cholesky': {'L': L}}
