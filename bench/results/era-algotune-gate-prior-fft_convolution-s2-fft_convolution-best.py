import numpy as np
from scipy.fft import next_fast_len

def solve(problem):
    x = problem["signal_x"]
    y = problem["signal_y"]
    mode = problem.get("mode", "full")
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    n, m = len(x), len(y)
    
    # For very small sizes, direct convolution is faster
    if n * m < 4096:
        if mode == "full":
            result = np.convolve(x, y)
        elif mode == "same":
            result = np.convolve(x, y, mode="same")
        else:
            result = np.convolve(x, y, mode="valid")
        return {"convolution": result}
    
    fft_size = next_fast_len(n + m - 1)
    
    X = np.fft.rfft(x, fft_size)
    Y = np.fft.rfft(y, fft_size)
    conv = np.fft.irfft(X * Y, fft_size)[:n + m - 1]
    
    if mode == "full":
        result = conv
    elif mode == "same":
        start = (m - 1) // 2
        result = conv[start:start + n]
    else:  # valid
        length = abs(n - m) + 1
        start = min(n, m) - 1
        result = conv[start:start + length]
    
    return {"convolution": result}
