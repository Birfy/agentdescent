import numpy as np
from scipy.linalg import lu

def solve(problem):
    A = np.array(problem['matrix'], dtype=float, copy=False)
    P, L, U = lu(A, check_finite=False)
    return {'LU': {'P': P.tolist(), 'L': L.tolist(), 'U': U.tolist()}}
