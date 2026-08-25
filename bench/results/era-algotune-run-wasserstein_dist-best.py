import numpy as np

def solve(problem):
    u = problem['u']
    v = problem['v']
    diff = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    np.cumsum(diff, out=diff)
    return float(np.abs(diff).sum())
