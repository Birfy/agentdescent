import numpy as np
from scipy.signal import fftconvolve

def solve(problem):
    x = np.asarray(problem['signal_x'], dtype=float)
    y = np.asarray(problem['signal_y'], dtype=float)
    mode = problem.get('mode', 'full')

    # Handle empty inputs
    if x.size == 0 or y.size == 0:
        if mode == 'same':
            result = np.zeros(max(x.size, y.size))
        else:
            result = np.zeros(0)
        return {'convolution': result}

    # scipy's fftconvolve handles all three modes correctly and efficiently
    result = fftconvolve(x, y, mode=mode)
    return {'convolution': result}
