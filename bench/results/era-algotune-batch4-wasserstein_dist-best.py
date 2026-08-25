import numpy as np

def solve(problem):
    u = np.asarray(problem['u'], dtype=np.float64)
    v = np.asarray(problem['v'], dtype=np.float64)
    n = u.size
    if n == 0:
        return 0.0
    diff = u - v
    np.cumsum(diff, out=diff)
    return float(np.abs(diff).sum())
