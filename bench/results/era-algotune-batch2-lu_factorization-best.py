import numpy as np
from scipy.linalg import lu

def solve(problem):
    A = np.asarray(problem['matrix'], dtype=np.float64)
    P, L, U = lu(A)
    return {'LU': {'P': P.tolist(), 'L': L.tolist(), 'U': U.tolist()}}
