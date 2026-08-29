import numpy as np
from scipy.linalg import lu

def solve(problem):
    A = np.asarray(problem["matrix"], dtype=np.float64)
    P, L, U = lu(A, check_finite=False)
    return {
        "LU": {
            "P": P,
            "L": L,
            "U": U
        }
    }
