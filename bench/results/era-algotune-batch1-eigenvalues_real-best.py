import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

def solve(problem):
    """
    Solve the eigenvalues problem for the given symmetric matrix.
    Returns a list of eigenvalues in descending order.
    """
    a = np.ascontiguousarray(problem, dtype=np.float64)
    syevr = get_lapack_funcs('syevr', (a,))
    w, _, _, _, _ = syevr(a, compute_v=0, lower=1, range='A', il=1, iu=a.shape[0])
    return w[::-1].tolist()
