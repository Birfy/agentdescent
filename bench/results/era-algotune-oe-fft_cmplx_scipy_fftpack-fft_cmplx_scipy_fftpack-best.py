import numpy as np
from scipy.fft import fftn

def solve(problem):
    # Ensure contiguous complex128 array
    if isinstance(problem, np.ndarray) and problem.dtype == np.complex128 and problem.flags['C_CONTIGUOUS']:
        arr = problem
    else:
        arr = np.ascontiguousarray(problem, dtype=np.complex128)
    return fftn(arr, workers=-1, overwrite_x=True)
