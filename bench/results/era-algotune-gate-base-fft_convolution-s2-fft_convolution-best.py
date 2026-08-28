import numpy as np
from scipy import signal

def solve(problem):
    signal_x = np.asarray(problem['signal_x'], dtype=np.float64)
    signal_y = np.asarray(problem['signal_y'], dtype=np.float64)
    mode = problem.get('mode', 'full')
    
    convolution_result = signal.fftconvolve(signal_x, signal_y, mode=mode)
    
    return {'convolution': convolution_result}
