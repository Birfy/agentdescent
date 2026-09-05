import numpy as np
from scipy.linalg import lu as scipy_lu

def solve(problem):
    A = np.asarray(problem['matrix'])
    P, L, U = scipy_lu(A, check_finite=False)
    return {'LU': {'P': P, 'L': L, 'U': U}}
