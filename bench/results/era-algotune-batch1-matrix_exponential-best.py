import numpy as np
from scipy.linalg import expm

def solve(problem):
    A = np.ascontiguousarray(problem['matrix'], dtype=np.float64)
    expA = expm(A)
    return {'exponential': expA}
