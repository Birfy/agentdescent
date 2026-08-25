import numpy as np
from scipy.linalg.lapack import dsyevd

def solve(problem):
    A = np.ascontiguousarray(problem, dtype=np.float64)
    w, v, info = dsyevd(A, compute_v=1, lower=1)
    if info != 0:
        w, v = np.linalg.eigh(A)
    return (w[::-1].tolist(), v[:, ::-1].T.tolist())
