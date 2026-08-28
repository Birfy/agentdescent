import numpy as np
from scipy.fft import fftn as scipy_fftn

def solve(problem):
    arr = np.asarray(problem, dtype=np.complex128)
    # For n=1860, scipy's pocketfft with workers=-1 is the fastest known approach
    # Use in-place to minimize memory allocation overhead
    result = scipy_fftn(arr, workers=-1, overwrite_x=True)
    return result
