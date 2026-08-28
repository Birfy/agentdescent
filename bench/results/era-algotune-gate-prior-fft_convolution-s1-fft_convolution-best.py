import numpy as np
from scipy import signal

def solve(problem: dict[str, list | str]) -> dict[str, np.ndarray]:
    """
    Compute convolution of two signals using FFT.
    
    Args:
        problem: Dictionary with keys:
            - "signal_x": list of numbers
            - "signal_y": list of numbers
            - "mode": "full", "same", or "valid"
    
    Returns:
        Dictionary with "convolution" key containing numpy array result
    """
    signal_x = np.asarray(problem['signal_x'], dtype=np.float64)
    signal_y = np.asarray(problem['signal_y'], dtype=np.float64)
    mode = problem.get('mode', 'full')
    
    # Use scipy's optimized FFT convolution
    convolution_result = signal.fftconvolve(signal_x, signal_y, mode=mode)
    
    return {'convolution': convolution_result}
