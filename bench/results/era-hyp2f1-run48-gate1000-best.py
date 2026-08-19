import math
import cmath
import numpy as np
from scipy.special import hyp2f1 as _scipy_hyp2f1, gamma, gammaln, poch

def _is_nonpos_int(x, tol=1e-10):
    if x > tol: return False
    n = round(x)
    return n <= 0 and abs(x - n) < tol

def _scipy_call(a, b, c, z):
    try:
        val = complex(_scipy_hyp2f1(a, b, c, z))
        if math.isfinite(val.real) and math.isfinite(val.imag):
            return val.real
    except Exception:
        pass
    return None

def _taylor_sum(a, b, c, z, max_iter=10000):
    term = 1.0 + 0j
    val = 1.0 + 0j
    for n in range(1, max_iter):
        term *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * z
        val += term
        if abs(term) < 1e-18 * abs(val):
            break
    return val

def _safe_gamma(x):
    if _is_nonpos_int(x, 1e-12):
        return math.inf
    try:
        v = gamma(x)
        if math.isfinite(v): return v
    except Exception:
        pass
    return math.inf

def _safe_gammaln(x):
    if _is_nonpos_int(x, 1e-12):
        return math.inf
    try:
        v = gammaln(x)
        if math.isfinite(v): return v
    except Exception:
        pass
    return math.inf

def _eval_2f1(a, b, c, z):
    if abs(z) < 1e-15:
        return 1.0

    if _is_nonpos_int(c, 1e-12):
        return _scipy_call(a, b, c, z)

    az = abs(z)

    if az < 0.9:
        val = _taylor_sum(a, b, c, z)
        if math.isfinite(val.real):
            return val.real

    if az < 1.1 and z < 1.0:
        w = z / (z - 1/ 1.0)
        if abs(w) < 0.9:
            val = _taylor_sum(a, c - b, c, w)
            if math.isfinite(val.real):
                base = 1.0 - z
                if base > 0:
                    return float(base**(-a)) * val.real
                else:
                    return cmath.exp(-a * cmath.log(base)).real * val.real

    if az > 1+1.2:
        if abs(b - a) < 1e-5:
            if abs(c - a) < 1e-8:
                base = 1.0 - z
                if base > 0:
                    return float(base**(-a))
                else:
                    return cmath.exp(-a * cmath.log(base)).real
            else:
                w = z / (z - 1.0)
                if abs(w) < 0.9:
                    val = _taylor_sum(a, a, c, w)
                    if math.isfinite(val.real):
                        base = 1.0 - z
                        if base > 0:
                            return float(base**(-a)) * val.real
                        else:
                            return cmath.exp(-a * cmath.log(base)).real * val.real
                return _scipy_call(a, b, c, z)
        else:
            if _is_nonpos_int(b - a, 1e-10) or _is_nonpos_int(a - b, 1e-10):
                return _scipy_call(a, b, c, z)

            w = 1.0 / z
            Lz = cmath.log(-z)
            log_z = -a * Lz
            log_w = -b * Lz

            ga, gb, gc = _safe_gamma(a), _safe_gamma(b), _safe_gamma(c)
            gba = _safe_gamma(b - a)
            gab = _safe_gamma(a - b)
            gca = _safe_gamma(c - a)
            gcb = _safe_gamma(c - b)

            if any(math.isinf(x) for x in (ga, gb, gc, gba, gab, gca, gcb)):
                return _scipy_call(a, b, c, z)

            c1 = gc * gba / (gb * gca)
            c2 = gc * gab / (ga * gcb)

            v1 = _taylor_sum(a, 1.0 - c + a, 1.0 - b + a, w)
            v2 = _taylor_sum(b, 1.0 - c + b, 1.0 - a + b, w)

            if not (math.isfinite(v1.real) and math.isfinite(v1.imag) and
                    math.isfinite(v2.real) and math.isfinite(v2.imag)):
                return _scipy_call(a, b, c, z)

            t1 = c1 * cmath.exp(log_z) * v1
            t2 = c2 * cmath.exp(log_w) * v2
            res = t1 + t2

            if math.isfinite(res.real):
                return res.real

    return _scipy_call(a, b, c, z)

def _analytic_cont(a, b, c, z):
    if abs(b - a) < 1e-8:
        return _scipy_call(a, b, c, z)

    g_c = _safe_gamma(c)
    g_ba = _safe_gamma(b - a)
    g_ca = _safe_gamma(c - a)
    g_ab = _safe_gamma(a - b)
    g_cb = _safe_gamma(c - b)

    if any(math.isinf(x) for x in (g_c, g_ba, g_ca, g_ab, g_cb)):
        return _scipy_call(a, b, c, z)

    w = 1.0 - z
    Lw = cmath.log(w)

    v1 = _taylor_sum(a, b, 1.0 + a + b - c, 1.0 - z)
    v2 = _taylor_sum(c - a, c - b, 1.0 - a - b + c, 1.0 - z)

    if not (math.isfinite(v1.real) and math.isfinite(v1.imag) and
            math.isfinite(v2.real) and math.isfinite(v2.imag)):
        return _scipy_call(a, b, c, z)

    try:
        g_a = _safe_gamma(a)
        g_b = _safe_gamma(b)
        g_cab = _safe_gamma(c - a - b)
        g_1mc = _safe_gamma(1.0 - c)
        g_1pambmc = _safe_gamma(1.0 + a + b - c)
    except Exception:
        return _scipy_call(a, b, c, z)

    if any(math.isinf(x) for x in (g_a, g_b, g_cab, g_1mc, g_1pambmc)):
        return _scipy_call(a, b, c, z)

    c1 = g_c * g_cab / (g_a * g_b)
    c2 = g_c * g_1mc / (g_cab * g_1pambmc)

    t1 = c1 * v1
    t2 = c2 * cmath.exp((c - a - b) * Lw) * v2

    res = t1 + t2
    if math.isfinite(res.real):
        return res.real
    return _scipy_call(a, b, c, z)

def hyp2f1(a, b, c, z):
    try:
        val = _eval_2f1(a, b, c, z)
        if val is not None and math.isfinite(val):
            return val
    except Exception:
        pass

    try:
        val = _analytic_cont(a, b, c, z)
        if val is not None and math.isfinite(val):
            return val
    except Exception:
        pass

    try:
        val = _scipy_call(a, b, c, z)
        if val is not None and math.isfinite(val):
            return val
    except Exception:
        pass

    return 0.0
