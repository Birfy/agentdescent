import numpy as np
from scipy.linalg.lapack import dsyevd

def solve(problem):
    a = np.ascontiguousarray(problem, dtype=np.float64)
    w, v, info = dsyevd(a, lower=1, compute_v=1, overwrite_a=1)
    if info != 0:
        w, v = np.linalg.eigh(a)
    w = w[::-1]
    v = v.T[::-1]
    return (w.tolist(), v.tolist())
