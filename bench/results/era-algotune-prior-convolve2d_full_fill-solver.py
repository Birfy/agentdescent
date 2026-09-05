import numpy as np
from scipy.fft import next_fast_len

def solve(problem: tuple) -> np.ndarray:
    a, b = problem
    
    # Output shape for full convolution
    out_h = a.shape[0] + b.shape[0] - 1
    out_w = a.shape[1] + b.shape[1] - 1
    
    # Use next fast length (typically powers of 2, 3, 5) for better FFT performance
    fh = next_fast_len(out_h)
    fw = next_fast_len(out_w)
    
    # Perform FFT-based convolution
    fa = np.fft.rfft2(a, (fh, fw))
    fb = np.fft.rfft2(b, (fh, fw))
    
    conv = np.fft.irfft2(fa * fb, (fh, fw))
    
    return conv[:out_h, :out_w]
