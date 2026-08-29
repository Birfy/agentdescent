import numpy as np
from scipy import signal
from numba import njit, prange, set_num_threads
import numba

# Set thread count for numba
set_num_threads(4)

@njit(cache=True, fastmath=True, parallel=True)
def _direct_convolve(x, y, mode):
    """Numba-accelerated direct convolution for small/medium arrays"""
    nx, ny = len(x), len(y)
    n = nx + ny - 1
    
    if mode == 'full':
        start, end = 0, n
    elif mode == 'same':
        if nx >= ny:
            start = (ny - 1) // 2
            end = start + nx
        else:
            start = (nx - 1) // 2
            end = start + ny
    else:  # valid
        if nx >= ny:
            start = ny - 1
            end = nx
        else:
            start = nx - 1
            end = ny
        if end <= start:
            return np.zeros(0)
    
    out_len = end - start
    result = np.zeros(out_len, dtype=np.float64)
    
    for i in prange(out_len):
        idx = start + i
        s = 0.0
        lo = max(0, idx - ny + 1)
        hi = min(nx - 1, idx)
        for k in range(lo, hi + 1):
            s += x[k] * y[idx - k]
        result[i] = s
    return result

@njit(cache=True, fastmath=True)
def _direct_convolve_small(x, y, mode):
    """Non-parallel direct convolution for very small arrays (lower overhead)"""
    nx, ny = len(x), len(y)
    n = nx + ny - 1
    
    if mode == 'full':
        start, end = 0, n
    elif mode == 'same':
        if nx >= ny:
            start = (ny - 1) // 2
            end = start + nx
        else:
            start = (nx - 1) // 2
            end = start + ny
    else:  # valid
        if nx >= ny:
            start = ny - 1
            end = nx
        else:
            start = nx - 1
            end = ny
        if end <= start:
            return np.zeros(0)
    
    out_len = end - start
    result = np.zeros(out_len, dtype=np.float64)
    
    for i in range(out_len):
        idx = start + i
        s = 0.0
        lo = max(0, idx - ny + 1)
        hi = min(nx - 1, idx)
        for k in range(lo, hi + 1):
            s += x[k] * y[idx - k]
        result[i] = s
    return result

@njit(cache=True, fastmath=True)
def _fft_convolve_numba(x, y, mode):
    """Numba-accelerated FFT convolution using real FFT"""
    nx, ny = len(x), len(y)
    n = nx + ny - 1
    
    # Find next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < n:
        fft_size <<= 1
    
    # Pad arrays
    x_pad = np.zeros(fft_size, dtype=np.float64)
    y_pad = np.zeros(fft_size, dtype=np.float64)
    x_pad[:nx] = x
    y_pad[:ny] = y
    
    # FFT, multiply, inverse FFT
    X = np.fft.rfft(x_pad)
    Y = np.fft.rfft(y_pad)
    Z = X * Y
    result = np.fft.irfft(Z, n=fft_size)[:n]
    
    # Handle mode
    if mode == 'full':
        return result
    elif mode == 'same':
        if nx >= ny:
            start = (ny - 1) // 2
            end = start + nx
        else:
            start = (nx - 1) // 2
            end = start + ny
        return result[start:end]
    else:  # valid
        if nx >= ny:
            start = ny - 1
            end = nx
        else:
            start = nx - 1
            end = ny
        if end <= start:
            return np.zeros(0)
        return result[start:end]

def solve(problem: dict[str, list | str]) -> dict[str, np.ndarray]:
    """FFT convolution with adaptive strategy for maximum performance"""
    signal_x = problem['signal_x']
    signal_y = problem['signal_y']
    mode = problem.get('mode', 'full')
    
    # Convert to numpy arrays with float64 for precision
    x = np.asarray(signal_x, dtype=np.float64)
    y = np.asarray(signal_y, dtype=np.float64)
    
    nx, ny = len(x), len(y)
    
    # Adaptive strategy based on problem size
    if nx * ny < 2000:
        # Very small: use non-parallel direct convolution (less overhead)
        result = _direct_convolve_small(x, y, mode)
    elif nx * ny < 20000:
        # Small: use parallel direct convolution
        result = _direct_convolve(x, y, mode)
    else:
        # For large arrays, use FFT convolution
        # Try numba FFT first (might be faster for some sizes)
        try:
            result = _fft_convolve_numba(x, y, mode)
        except:
            # Fallback to scipy if numba FFT fails
            result = signal.fftconvolve(x, y, mode=mode)
    
    return {'convolution': result}

PROMISE: 7
