import numpy as np

def solve(problem):
    a, b = problem
    M1, N1 = a.shape
    M2, N2 = b.shape
    out_shape = (M1 + M2 - 1, N1 + N2 - 1)
    fft_rows = 1 << (out_shape[0] - 1).bit_length()
    fft_cols = 1 << (out_shape[1] - 1).bit_length()
    A = np.fft.rfft2(a, s=(fft_rows, fft_cols))
    B = np.fft.rfft2(b, s=(fft_rows, fft_cols))
    result = np.fft.irfft2(A * B, s=(fft_rows, fft_cols))
    return result[:out_shape[0], :out_shape[1]]
