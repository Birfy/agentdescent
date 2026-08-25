import numpy as np
from scipy.linalg import svd as scipy_svd

def solve(problem):
    A = problem['matrix']
    U, s, Vh = scipy_svd(A, full_matrices=False, lapack_driver='gesdd', overwrite_a=False, check_finite=False)
    return {'U': U, 'S': s, 'V': Vh.T}
