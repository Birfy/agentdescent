import numpy as np
from scipy.fft import next_fast_len

def solve(problem: tuple) -> np.ndarray:
    a, b = problem
    
    # Full convolution output size
    out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    
    # Use FFT with a fast length (not necessarily power of 2) for efficiency
    fft_shape = (next_fast_len(out_shape[0]), next_fast_len(out_shape[1]))
    
    # FFT both, multiply, inverse
    A = np.fft.rfft2(a, s=fft_shape)
    B = np.fft.rfft2(b, s=fft_shape)
    conv = np.fft.irfft2(A * B, s=fft_shape)
    
    # Crop to full size
    return conv[:out_shape[0], :out_shape[1]]

PROMISE: 10
