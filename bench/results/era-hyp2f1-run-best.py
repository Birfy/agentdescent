import math
import cmath
import numpy as np
from scipy.special import hyp2f1 as _scipy_hyp2f1, gammaln, loggamma, gamma, digamma

def _calc_log_gamma_sign(x):
    if x == math.floor(x) and x <= 0:
        return -float('inf'), 1.0
    if x > 0:
        return gammaln(x), 1.0
    else:
        v = loggamma(x)
        if math.isinf(v.real) or math.isnan(v.real):
            return -float('inf'), 1.0
        return v.real, -1.0 if abs(v.imag) > 1e-12 else 1.0

def _log_abs(x):
    return math.log(abs(x)) if x != 0.0 else -float('inf')

def _log_sum_exp(log_A, sign_A, log_B, sign_B):
    if log_A == -float('inf') and log_B == -float('inf'):
        return 0.0
    if log_A > log_B:
        return sign_A * math.exp(log_A) * (1.0 + sign_B / sign_A * math.exp(log_B - log_A))
    else:
        return sign_B * math.exp(log_B) * (1.0 + sign_A / sign_B * math.exp(log_A - log_B))

def _series_2f1(a, b, c, z, max_iter=200000):
    if abs(z) < 1e-12:
        return 1.0
    term = 1.0
    s = 1.0
    for n in range(1, max_iter):
        denom = c + n - 1.0
        if abs(denom) < 1e-15:
            continue
        term *= (a + n - 1.0) * (b + n - 1.0) * z / (n * denom)
        s += term
        if abs(term) < 1e-18 * abs(s) and n > 10:
            break
    return s

def _eval_2f1_series(a, b, c, z):
    if abs(z) < 0.9:
        return _series_2f1(a, b, c, z)
    return float('nan')

def _hyp2f1_neg_z(a, b, c, z):
    w = 1.0 / z
    
    grow1 = abs((a - c + 1.0) * (b - c + 1.0) * w)
    grow2 = abs(a * b * w)
    
    if grow1 > 0.5 and grow2 > 0.5:
        z_t = z / (z - 1.0)
        if abs(z_t) < 0.7:
            val = _hyp2f1_neg_z(a, c - b, c, z_t)
            if math.isfinite(val) and val != 0.0:
                if z < 1.0:
                    log_factor = -a * math.log1p(-z)
                    if abs(log_factor) < 700:
                        return math.exp(log_factor) * val
            return float('nan')

    use_f1 = grow1 <= grow2

    if use_f1:
        a1, b1, c1 = a - c + 1.0, b - c + 1.0, 2.0 - c
        a2, b2, c2 = a, b, a + b - c + 1.0
    else:
        a1, b1, c1 = a, b, a + b - c + 1.0
        a2, b2, c2 = a - c + 1.0, b - c + 1.0, 2.0 - c

    f1_val = _eval_2f1_series(a1, b1, c1, w)
    if not math.isfinite(f1_val):
        return float('nan')

    lg_c_a, s_c_a = _calc_log_gamma_sign(c - a)
    lg_c_b, s_c_b = _calc_log_gamma_sign(c - b)
    lg_c, s_c = _calc_log_gamma_sign(c)
    lg_c_ab, s_c_ab = _calc_log_gamma_sign(c - a - b)
    lg_a1, s_a1 = _calc_log_gamma_sign(a1)
    lg_b1, s_b1 = _calc_log_gamma_sign(b1)
    lg_c1, s_c1 = _calc_log_gamma_sign(c1)
    lg_a2, s_a2 = _calc_log_gamma_sign(a2)
    lg_b2, s_b2 = _calc_log_gamma_sign(b2)
    lg_c2, s_c2 = _calc_log_gamma_sign(c2)

    log_T1 = lg_c_a + lg_c_b - lg_c - lg_c_ab
    sign_T1 = s_c_a * s_c_b / (s_c * s_c_ab)

    log_T2 = lg_a1 + lg_b1 - lg_c1 - lg_c2
    sign_T2 = s_a1 * s_b1 / (s_c1 * s_c2)

    if w > 0:
        log_T1 -= w
    else:
        log_T2 += w

    if log_T2 == -float('inf'):
        if log_T1 > 700:
            return float('nan')
        return sign_T1 * math.exp(log_T1) * f1_val

    f2_val = _eval_2f1_series(a2, b2, c2, w)
    if not math.isfinite(f2_val):
        return float('nan')

    log_A = log_T1 + _log_abs(f1_val)
    sign_A = sign_T1 * (1.0 if f1_val > 0 else -1.0)

    log_B = log_T2 + _log_abs(f2_val)
    sign_B = sign_T2 * (1.0 if f2_val > 0 else -1.0)

    if log_A > 700 or log_B > 700:
        return float('nan')

    return _log_sum_exp(log_A, sign_A, log_B, sign_B)

def _hyp2f1_near_one(a, b, c, z):
    z1 = 1.0 - z
    if z1 <= 0:
        return float('nan')
    log_z1 = math.log(z1)
    
    grow1 = abs(a * b * z1)
    grow2 = abs((c - a) * (c - b) * z1)
    
    if grow1 > 0.5 and grow2 > 0.5:
        z_t = -z1 / z
        if abs(z_t) < 0.7:
            val = _hyp2f1_neg_z(c - a, c - b, c, z_t)
            if math.isfinite(val) and val != 0.0:
                log_factor = (c - a - b) * log_z1
                if abs(log_factor) < 700:
                    return math.exp(log_factor) * val
        return float('nan')

    use_f1 = grow1 <= grow2

    if use_f1:
        a1, b1, c1 = a, b, a + b - c + 1.0
        a2, b2, c2 = c - a, c - b, c - a - b + 1.0
        p = c - a - b
    else:
        a1, b1, c1 = c - a, c - b, c - a - b + 1.0
        a2, b2, c2 = a, b, a + b - c + 1.0
        p = a + b - c

    f1_val = _eval_2f1_series(a1, b1, c1, z1)
    if not math.isfinite(f1_val):
        return float('nan')

    lg_c_a, s_c_a = _calc_log_gamma_sign(c - a)
    lg_c_b, s_c_b = _calc_log_gamma_sign(c - b)
    lg_c, s_c = _calc_log_gamma_sign(c)
    lg_ab, s_ab = _calc_log_gamma_sign(a + b - c)
    lg_a1, s_a1 = _calc_log_gamma_sign(a1)
    lg_b1, s_b1 = _calc_log_gamma_sign(b1)
    lg_c1, s_c1 = _calc_log_gamma_sign(c1)
    lg_a2, s_a2 = _calc_log_gamma_sign(a2)
    lg_b2, s_b2 = _calc_log_gamma_sign(b2)
    lg_c2, s_c2 = _calc_log_gamma_sign(c2)

    log_T1 = lg_c_a + lg_c_b - lg_c - lg_ab
    sign_T1 = s_c_a * s_c_b / (s_c * s_ab)

    log_T2 = lg_a1 + lg_b1 - lg_c1 - lg_c2
    sign_T2 = s_a1 * s_b1 / (s_c1 * s_c2)

    if p > 0:
        log_T1 += p * log_z1
    else:
        log_T2 -= p * log_z1

    if log_T2 == -float('inf'):
        if log_T1 > 700:
            return float('nan')
        return sign_T1 * math.exp(log_T1) * f1_val

    f2_val = _eval_2f1_series(a2, b2, c2, z1)
    if not math.isfinite(f2_val):
        return float('nan')

    log_A = log_T1 + _log_abs(f1_val)
    sign_A = sign_T1 * (1.0 if f1_val > 0 else -1.0)

    log_B = log_T2 + _log_abs(f2_val)
    sign_B = sign_T2 * (1.0 if f2_val > 0 else -1.0)

    if log_A > 700 or log_B > 700:
        return float('nan')

    return _log_sum_exp(log_A, sign_A, log_B, sign_B)

def hyp2f1(a, b, c, z):
    try:
        a = float(a); b = float(b); c = float(c); z = float(z)
    except Exception:
        return float('nan')

    if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(c) or not math.isfinite(z):
        return float('nan')

    if z == 0.0:
        return 1.0

    if abs(z) < 0.9:
        val = _series_2f1(a, b, c, z)
        if math.isfinite(val):
            return val

    if z < -0.1:
        val = _hyp2f1_neg_z(a, b, c, z)
        if math.isfinite(val) and val != 0.0:
            return val

    if abs(z - 1.0) < 0.5:
        val = _hyp2f1_near_one(a, b, c, z)
        if math.isfinite(val) and val != 0.0:
            return val

    try:
        val = float(_scipy_hyp2f1(a, b, c, z))
        if math.isfinite(val):
            return val
    except Exception:
        pass

    if z < -0.1:
        val = _hyp2f1_neg_z(a, b, c, z)
        if math.isfinite(val) and val != 0.0:
            return val

    return 0.0
