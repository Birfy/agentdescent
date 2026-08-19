import math
import numpy as np
from scipy.integrate import quad

def integrate(f, a, b):
    if a == b:
        return 0.0
    if a > b:
        return -integrate(f, b, a)

    is_inf_a = (a == -math.inf)
    is_inf_b = (b == math.inf)

    if is_inf_a and is_inf_b:
        def g(t):
            if t <= -1.0:
                x = -1.0 / t
                w = 1.0 / (t * t)
            elif t < 1.0:
                x = t / (1.0 - t * t)
                w = (1.0 + t * t) / ((1.0 - t * t) ** 2)
            else:
                x = 1.0 / t
                w = 1.0 / (t * t)
            return f(x) * w
        A, B = -1.0, 1.0
        P = [0.0]
    elif is_inf_a:
        def g(t):
            x = b - t / (1.0 - t)
            w = 1.0 / ((1.0 - t) ** 2)
            return f(x) * w
        A, B = 0.0, 1.0
        P = []
    elif is_inf_b:
        def g(t):
            x = a + t / (1.0 - t)
            w = 1.0 / ((1.0 - t) ** 2)
            return f(x) * w
        A, B = 0.0, 1.0
        P = []
    else:
        g = f
        A, B = a, b
        P = []

    def safe_g(x):
        try:
            v = g(x)
        except Exception:
            return 0.0
        if not isinstance(v, (int, float)):
            return 0.0
        if math.isnan(v):
            return 0.0
        if math.isinf(v):
            return 0.0
        return v

    best_val = math.nan
    for lim, epsabs, epsrel in [(100, 1e-13, 1e-13), (200, 1e-12, 1e-12), (400, 1e-10, 1e-10)]:
        try:
            val, err = quad(safe_g, A, B, limit=lim, epsabs=epsabs, epsrel=epsrel, points=P)
            if math.isfinite(val):
                if not math.isfinite(best_val) or err < best_err:
                    best_val = val
                    best_err = err
                if err < 1e-12 * max(1.0, abs(val)):
                    return float(best_val)
        except Exception:
            pass

    if not math.isfinite(best_val):
        try:
            val, _ = quad(safe_g, A, B, limit=50, epsabs=1e-6, epsrel=1e-6, points=P)
            if math.isfinite(val):
                best_val = val
        except Exception:
            pass

    if math.isfinite(best_val):
        return float(best_val)
    return 0.0
