import numpy as np
from scipy.fft import next_fast_len

def solve(problem: tuple) -> np.ndarray:
    a, b = problem
    
    # Output size for full convolution
    out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    
    # Use next fast FFT length (supports 2, 3, 5, 7 for scipy's pocketfft)
    fft_shape = (next_fast_len(out_shape[0]), next_fast_len(out_shape[1]))
    
    # Compute FFTs and multiply
    A = np.fft.rfft2(a, s=fft_shape)
    B = np.fft.rfft2(b, s=fft_shape)
    C = A * B
    
    # Inverse FFT and crop
    result = np.fft.irfft2(C, s=fft_shape)
    result = result[:out_shape[0], :out_shape[1]]
    
    return result
