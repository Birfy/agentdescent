import logging
import random
from enum import Enum
from typing import Any
import numpy as np
from scipy import signal

def solve(problem: dict[str, Any]) -> dict[str, list]:
    """
        Solve the convolution problem using the Fast Fourier Transform approach.

        Uses scipy.signal.fftconvolve to compute the convolution of signals x and y.

        :param problem: A dictionary representing the convolution problem.
        :return: A dictionary with key:
                 "convolution": a list representing the convolution result.
        """
    signal_x = np.array(problem['signal_x'])
    signal_y = np.array(problem['signal_y'])
    mode = problem.get('mode', 'full')
    convolution_result = signal.fftconvolve(signal_x, signal_y, mode=mode)
    solution = {'convolution': convolution_result}
    return solution
