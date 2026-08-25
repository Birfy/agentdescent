import logging
import numpy as np
from scipy import signal
_REF_MODE = 'full'

def solve(problem: tuple) -> np.ndarray:
    a, b = problem
    return signal.convolve(a, b, mode=_REF_MODE)
