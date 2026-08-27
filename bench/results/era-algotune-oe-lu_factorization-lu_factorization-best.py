import numpy as np

def solve(problem):
    A = np.asarray(problem['matrix'], dtype=np.float64)
    n = A.shape[0]
    
    # Use scipy's lu for reliable partial pivoting
    from scipy.linalg import lu
    P, L, U = lu(A, check_finite=False)
    
    return {'LU': {'P': P.tolist(), 'L': L.tolist(), 'U': U.tolist()}}
