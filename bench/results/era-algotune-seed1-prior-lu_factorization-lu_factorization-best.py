import numpy as np
from scipy.linalg import lu as scipy_lu

def solve(problem):
    A = np.array(problem['matrix'], dtype=np.float64, order='C')
    P, L, U = scipy_lu(A, check_finite=False, overwrite_a=True)
    return {'LU': {'P': P.tolist(), 'L': L.tolist(), 'U': U.tolist()}}
