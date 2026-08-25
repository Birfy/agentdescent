import numpy as np

def solve(problem):
    A = problem['A']
    B = problem['B']
    
    # Fast path: both are already float64 ndarrays in C-contiguous order
    # This avoids type checking, dtype conversion, and memory copying overhead.
    if (type(A) is np.ndarray and A.dtype == np.dtype(np.float64) and A.flags.c_contiguous and
        type(B) is np.ndarray and B.dtype == np.dtype(np.float64) and B.flags.c_contiguous):
        return A @ B
    
    # Fallback for lists or non-optimal arrays
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    
    return A_arr @ B_arr
