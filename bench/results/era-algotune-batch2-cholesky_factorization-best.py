import numpy as np
import scipy.linalg

def solve(problem):
    A = problem['matrix']
    L = scipy.linalg.cholesky(A, lower=True, check_finite=False, overwrite_a=False)
    return {'Cholesky': {'L': L}}
