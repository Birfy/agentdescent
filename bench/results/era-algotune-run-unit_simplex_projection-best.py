import numpy as np

def solve(problem):
    y = np.asarray(problem['y'], dtype=np.float64)
    n = y.size
    
    if n == 0:
        return {"solution": np.array([], dtype=np.float64)}
    
    # Sort y in descending order. np.sort is ascending, so we slice with [::-1].
    # This is faster than np.sort(y)[::-1] as it combines the operations.
    u = np.sort(y)[::-1]
    
    # Compute cumulative sum and subtract 1
    cssv = np.cumsum(u) - 1.0
    
    # Find rho: the largest index i such that u[i] > cssv[i] / (i + 1)
    # This is equivalent to: u[i] * (i + 1) > cssv[i]
    # We compute this without creating explicit index arrays.
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    
    # Compute threshold theta
    theta = cssv[rho] / (rho + 1)
    
    # Compute projection
    x = np.maximum(y - theta, 0.0)
    
    return {"solution": x}
