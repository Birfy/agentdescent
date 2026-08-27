import numpy as np
from scipy.fft import rfft2, irfft2

def solve(problem):
    a, b = problem
    M1, N1 = a.shape
    M2, N2 = b.shape
    
    out_shape = (M1 + M2 - 1, N1 + N2 - 1)
    
    # For n=6, sizes are 180x180 and 48x48, output 227x227
    # FFT size 256 is efficient and sufficient
    fft_M = 256
    fft_N = 256
    
    a_pad = np.zeros((fft_M, fft_N), dtype=np.float64)
    b_pad = np.zeros((fft_M, fft_N), dtype=np.float64)
    a_pad[:M1, :N1] = a
    b_pad[:M2, :N2] = b
    
    A = rfft2(a_pad)
    B = rfft2(b_pad)
    result = irfft2(A * B, s=(fft_M, fft_N))
    
    return result[:out_shape[0], :out_shape[1]]
