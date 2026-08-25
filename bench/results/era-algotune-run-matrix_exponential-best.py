import numpy as np
from scipy.linalg import expm

def solve(problem):
    A = np.asarray(problem['matrix'], dtype=np.float64)
    return {'exponential': expm(A)}
