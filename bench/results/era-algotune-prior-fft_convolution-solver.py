import numpy as np

def solve(problem):
    x = np.asarray(problem["signal_x"], dtype=np.float64)
    y = np.asarray(problem["signal_y"], dtype=np.float64)
    mode = problem.get("mode", "full")

    n, m = len(x), len(y)

    if n == 0 or m == 0:
        if mode == "same":
            result = np.zeros(max(n, m))
        else:
            result = np.zeros(0)
        return {"convolution": result}

    # Use scipy's FFT-based convolution for speed and correctness
    from scipy.signal import fftconvolve

    if mode == "full":
        result = fftconvolve(x, y, mode="full")
    elif mode == "same":
        result = fftconvolve(x, y, mode="same")
    else:  # valid
        result = fftconvolve(x, y, mode="valid")

    return {"convolution": result}
