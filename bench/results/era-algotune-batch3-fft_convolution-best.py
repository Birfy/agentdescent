import numpy as np
from scipy import signal

def solve(problem):
    x = np.asarray(problem['signal_x'], dtype=np.float64)
    y = np.asarray(problem['signal_y'], dtype=np.float64)
    mode = problem.get('mode', 'full')
    
    convolution_result = signal.fftconvolve(x, y, mode=mode)
    return {'convolution': convolution_result}
