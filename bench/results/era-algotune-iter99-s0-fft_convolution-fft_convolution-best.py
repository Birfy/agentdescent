import numpy as np
from scipy import signal
import numba
from numba import prange

# Pre-compile the FFT convolution path for common sizes
@numba.njit(cache=True, fastmath=True)
def _fft_convolve_numba(x, y):
    """Numba-accelerated FFT convolution for real inputs."""
    n = len(x) + len(y) - 1
    # Use next power of 2 for FFT efficiency
    nfft = 1
    while nfft < n:
        nfft <<= 1
    
    # Pad and FFT
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    
    # Multiply in frequency domain
    Z = X * Y
    
    # Inverse FFT
    result = np.fft.irfft(Z, nfft)[:n]
    return result

def solve(problem: dict) -> dict:
    """
    Compute convolution using FFT approach with optimized implementation.
    
    Args:
        problem: dict with keys 'signal_x', 'signal_y', and optional 'mode'
        
    Returns:
        dict with 'convolution' key containing numpy array result
    """
    x = problem['signal_x']
    y = problem['signal_y']
    mode = problem.get('mode', 'full')
    
    # Convert to numpy arrays efficiently
    if isinstance(x, np.ndarray):
        signal_x = x
    else:
        signal_x = np.asarray(x, dtype=np.float64)
    
    if isinstance(y, np.ndarray):
        signal_y = y
    else:
        signal_y = np.asarray(y, dtype=np.float64)
    
    # Ensure float64 for optimal FFT performance
    if signal_x.dtype != np.float64:
        signal_x = signal_x.astype(np.float64)
    if signal_y.dtype != np.float64:
        signal_y = signal_y.astype(np.float64)
    
    # Ensure 1D arrays
    if signal_x.ndim != 1:
        signal_x = signal_x.ravel()
    if signal_y.ndim != 1:
        signal_y = signal_y.ravel()
    
    # Handle edge cases
    if len(signal_x) == 0 or len(signal_y) == 0:
        if mode == 'full':
            return {'convolution': np.zeros(0, dtype=np.float64)}
        elif mode == 'same':
            return {'convolution': np.zeros(max(len(signal_x), len(signal_y)), dtype=np.float64)}
        else:  # valid
            return {'convolution': np.zeros(0, dtype=np.float64)}
    
    # Use scipy's optimized FFT convolution for general cases
    # but with pre-allocated output buffer for speed
    if mode == 'full':
        # Direct FFT convolution for full mode
        n = len(signal_x) + len(signal_y) - 1
        nfft = 1 << (n - 1).bit_length()
        
        # Use rfft for real inputs (2x faster than complex FFT)
        X = np.fft.rfft(signal_x, nfft)
        Y = np.fft.rfft(signal_y, nfft)
        Z = X * Y
        conv = np.fft.irfft(Z, nfft)[:n]
    else:
        # For 'same' and 'valid' modes, use scipy's optimized implementation
        conv = signal.fftconvolve(signal_x, signal_y, mode=mode)
    
    return {'convolution': conv}
