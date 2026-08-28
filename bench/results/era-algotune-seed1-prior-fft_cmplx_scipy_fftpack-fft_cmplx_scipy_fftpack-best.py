import numpy as np
from scipy.fft import fft2 as scipy_fft2

def solve(problem):
    # Convert to contiguous complex128 array if needed
    if isinstance(problem, np.ndarray) and problem.dtype == np.complex128 and problem.flags['C_CONTIGUOUS']:
        arr = problem
    else:
        arr = np.ascontiguousarray(problem, dtype=np.complex128)
    
    # Use scipy's FFT with explicit thread control
    # scipy.fft.fft2 with workers=-1 uses all available cores
    # overwrite_x=True avoids unnecessary copies
    return scipy_fft2(arr, workers=-1, overwrite_x=True)

# Alternative: try to use pocketfft directly via scipy's low-level interface
# This can sometimes be faster than the high-level wrapper
try:
    from scipy.fft._pocketfft import pfft
    # Check if pfft is a module or a function
    if hasattr(pfft, 'pfft2'):
        _use_pocketfft = True
    else:
        _use_pocketfft = False
except ImportError:
    _use_pocketfft = False

def solve_optimized(problem):
    if isinstance(problem, np.ndarray) and problem.dtype == np.complex128 and problem.flags['C_CONTIGUOUS']:
        arr = problem
    else:
        arr = np.ascontiguousarray(problem, dtype=np.complex128)
    
    if _use_pocketfft:
        # Direct pocketfft call with 2D transform
        # pfft is a module, so access the pfft2 function
        return pfft.pfft2(arr, axes=(-2, -1), workers=-1, overwrite_x=True)
    else:
        return scipy_fft2(arr, workers=-1, overwrite_x=True)

# Use the optimized version as the main solve function
solve = solve_optimized
