import numpy as np

def solve(problem):
    y = np.array(problem['y'], dtype=np.float64).ravel()
    n = y.shape[0]
    
    # Sort y in descending order
    sorted_y = np.sort(y)[::-1]
    
    # Compute cumulative sum and subtract 1
    cumsum_y = np.cumsum(sorted_y) - 1.0
    
    # Find the largest index rho such that sorted_y[rho] > cumsum_y[rho] / (rho + 1)
    rho = np.nonzero(sorted_y * np.arange(1, n + 1, dtype=np.float64) > cumsum_y)[0][-1]
    
    # Compute the threshold theta
    theta = cumsum_y[rho] / (rho + 1)
    
    # Compute the projection
    x = np.maximum(y - theta, 0.0)
    
    return {'solution': x}
