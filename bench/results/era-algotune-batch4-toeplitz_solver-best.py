import numpy as np
from scipy.linalg import solve_toeplitz

def solve(problem):
    c = np.asarray(problem['c'], dtype=np.float64)
    r = np.asarray(problem['r'], dtype=np.float64)
    b = np.asarray(problem['b'], dtype=np.float64)
    
    x = solve_toeplitz((c, r), b)
    return x.tolist()
