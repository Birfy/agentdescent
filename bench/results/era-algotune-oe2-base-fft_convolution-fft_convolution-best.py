import numpy as np
from scipy import signal

def solve(problem: dict) -> dict:
    signal_x = problem['signal_x']
    signal_y = problem['signal_y']
    mode = problem.get('mode', 'full')

    # Convert to numpy arrays if needed
    if not isinstance(signal_x, np.ndarray):
        signal_x = np.asarray(signal_x, dtype=np.float64)
    else:
        signal_x = signal_x.astype(np.float64, copy=False)
    if not isinstance(signal_y, np.ndarray):
        signal_y = np.asarray(signal_y, dtype=np.float64)
    else:
        signal_y = signal_y.astype(np.float64, copy=False)

    # For full mode, use manual FFT to avoid scipy overhead
    if mode == 'full':
        n = len(signal_x) + len(signal_y) - 1
        # Compute next power of two for speed
        fft_size = 1 << (n - 1).bit_length()
        X = np.fft.rfft(signal_x, fft_size)
        Y = np.fft.rfft(signal_y, fft_size)
        Z = X * Y
        result = np.fft.irfft(Z, fft_size)[:n]
    else:
        # For 'same' and 'valid', scipy handles the indexing correctly
        result = signal.fftconvolve(signal_x, signal_y, mode=mode)

    return {'convolution': result}
