import numpy as np
from scipy import signal

def solve(problem):
    a, b = problem
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        return signal.convolve(a_arr, b_arr, mode='full')
    
    na = a_arr.size
    nb = b_arr.size
    
    if na == 0 or nb == 0:
        return np.convolve(a_arr, b_arr, mode='full')
    
    if na < 8 or nb < 8:
        return np.convolve(a_arr, b_arr, mode='full')
    
    if na <= 64 and nb <= 64:
        return signal.convolve(a_arr, b_arr, mode='full', method='direct')
    
    return signal.convolve(a_arr, b_arr, mode='full', method='fft')
