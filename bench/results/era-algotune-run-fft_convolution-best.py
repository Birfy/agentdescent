import numpy as np
from scipy import signal

def solve(problem):
    x = np.asarray(problem['signal_x'], dtype=np.float64)
    y = np.asarray(problem['signal_y'], dtype=np.float64)
    mode = problem.get('mode', 'full')
    return {'convolution': signal.fftconvolve(x, y, mode=mode)}
