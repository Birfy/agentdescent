import numpy as np

def solve(problem):
    u = problem['u']
    v = problem['v']
    
    if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
        if u.dtype == v.dtype and u.flags.c_contiguous and v.flags.c_contiguous:
            d = u - v
        else:
            d = np.ascontiguousarray(u) - np.ascontiguousarray(v)
    else:
        d = np.ascontiguousarray(u, dtype=np.float64) - np.ascontiguousarray(v, dtype=np.float64)
            
    np.cumsum(d, out=d)
    np.abs(d, out=d)
    return float(d.sum())
