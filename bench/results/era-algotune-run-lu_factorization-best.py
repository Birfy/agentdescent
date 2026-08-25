import numpy as np
from scipy.linalg import lu

def solve(problem):
    A = np.array(problem['matrix'], dtype=np.float64, copy=False)
    P, L, U = lu(A)
    return {'LU': {'P': P.tolist(), 'L': L.tolist(), 'U': U.tolist()}}
