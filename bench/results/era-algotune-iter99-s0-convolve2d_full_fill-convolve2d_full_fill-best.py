import numpy as np
from scipy.fft import next_fast_len
from numba import njit, prange

def solve(problem: tuple) -> np.ndarray:
    a, b = problem
    
    m1, n1 = a.shape
    m2, n2 = b.shape
    
    s1 = m1 + m2 - 1
    s2 = n1 + n2 - 1
    
    # For n=6: a is 180x180, b is 48x48
    # Direct convolution would be 180*180*48*48 ≈ 74.6M operations
    # FFT is much faster
    
    # Use next fast length (often better than power of 2 for real FFTs)
    fft_m = next_fast_len(s1, real=True)
    fft_n = next_fast_len(s2, real=True)
    
    # Real FFT for speed - only compute half the spectrum
    A = np.fft.rfft2(a, s=(fft_m, fft_n))
    B = np.fft.rfft2(b, s=(fft_m, fft_n))
    
    # Multiply in frequency domain
    result = np.fft.irfft2(A * B, s=(fft_m, fft_n))
    
    # Crop to the full convolution size
    return result[:s1, :s2]

PROMISE: 10
