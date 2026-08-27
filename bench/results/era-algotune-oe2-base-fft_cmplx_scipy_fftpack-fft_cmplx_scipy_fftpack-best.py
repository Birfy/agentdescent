import numpy as np
from scipy.fft import fftn as scipy_fftn

def solve(problem):
    # Fast path: if already complex128 and C-contiguous, avoid copy
    if isinstance(problem, np.ndarray) and problem.dtype == np.complex128 and problem.flags['C_CONTIGUOUS']:
        arr = problem
    else:
        arr = np.ascontiguousarray(problem, dtype=np.complex128)
    return scipy_fftn(arr, workers=-1, overwrite_x=True)
