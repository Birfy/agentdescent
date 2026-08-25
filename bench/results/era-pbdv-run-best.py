import math
import numpy as np
from scipy.special import pbdv as _scipy_pbdv
from scipy.special import hyperu, hyp1f1, gamma, gammaln

def pbdv(v, x):
    try:
        v = float(v)
        x = float(x)
    except Exception:
        return float('nan')

    if not math.isfinite(v) or not math.isfinite(x):
        return float('nan')

    if x == 0.0:
        if v == 0.0:
            return 1.0
        try:
            return float(gamma(0.5) / gamma((1.0 - v) / 2.0) / (2.0 ** (-v / 2.0)))
        except Exception:
            return float('nan')

    abs_x = abs(x)

    if x > 0:
        if v >= 15.0 and x >= 15.0:
            try:
                z = x * x / 2.0
                U = float(hyperu(-v / 2.0, 0.5, z))
                log_scale = (v / 2.0) * math.log(2.0) - x * x / 4.0
                if log_scale > 600.0:
                    if U == 0.0: return 0.0
                    logU = math.log(abs(U))
                    s = math.copysign(1.0, U)
                    log_val = log_scale + logU
                    if log_val > 700.0: return float('inf') * s
                    if log_val < -750.0: return 0.0
                    return s * math.exp(log_val)
                return float((2.0 ** (v / 2.0)) * math.exp(-x * x / 4.0) * U)
            except Exception:
                pass
        try:
            return float(_scipy_pbdv(v, x)[0])
        except Exception:
            pass
    else:
        if v >= 15.0 and abs_x >= 8.0:
            try:
                z = abs_x * abs_x / 2.0
                U = float(hyperu(-v / 2.0, 0.5, z))
                log_scale = (v / 2.0) * math.log(2.0) - abs_x * abs_x / 4.0
                if log_scale > 600.0:
                    if U == 0.0:
                        Dp = 0.0
                    else:
                        logU = math.log(abs(U))
                        s = math.copysign(1.0, U)
                        log_val = log_scale + logU
                        if log_val > 700.0: Dp = float('inf') * s
                        elif log_val < -750.0: Dp = 0.0
                        else: Dp = s * math.exp(log_val)
                else:
                    Dp = float((2.0 ** (v / 2.0)) * math.exp(-abs_x * abs_x / 4.0) * U)

                if Dp == 0.0 or not math.isfinite(Dp):
                    Dp = float(_scipy_pbdv(v, abs_x)[0])

                v2 = -v - 1.0
                U2 = float(hyperu(-v2 / 2.0, 0.5, z))
                log_scale2 = (v2 / 2.0) * math.log(2.0) - abs_x * abs_x / 4.0
                if log_scale2 > 600.0:
                    if U2 == 0.0:
                        Dm = 0.0
                    else:
                        logU2 = math.log(abs(U2))
                        s2 = math.copysign(1.0, U2)
                        log_val2 = log_scale2 + logU2
                        if log_val2 > 700.0: Dm = float('inf') * s2
                        elif log_val2 < -750.0: Dm = 0.0
                        else: Dm = s2 * math.exp(log_val2)
                else:
                    Dm = float((2.0 ** (v2 / 2.0)) * math.exp(-abs_x * abs_x / 4.0) * U2)

                if Dm == 0.0 or not math.isfinite(Dm):
                    Dm = float(_scipy_pbdv(v2, abs_x)[0])

                log_G = float(gammaln(-v))
                log_K = 0.5 * math.log(2.0 * math.pi) - log_G
                if log_K > 700.0:
                    return float('inf')
                K = math.exp(log_K)
                return float(math.cos(math.pi * v) * Dp - math.sin(math.pi * v) * K * Dm)
            except Exception:
                pass
        try:
            return float(_scipy_pbdv(v, x)[0])
        except Exception:
            pass

    try:
        return float(_scipy_pbdv(v, x)[0])
    except Exception:
        return float('nan')
