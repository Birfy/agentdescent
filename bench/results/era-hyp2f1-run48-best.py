import math
import numpy as np
from scipy.special import hyp2f1 as _scipy_hyp2f1, gammaln, gamma, digamma, poch

def _safe_scipy_hyp2f1(a, b, c, z):
    try:
        val = _scipy_hyp2f1(a, b, c, z)
        if math.isfinite(val):
            return float(val)
    except Exception:
        pass
    return None

def _is_near_nonpos_int(x, tol=1e-7):
    r = round(x)
    return abs(x - r) < tol and r <= 0

def _gamma_val(x):
    if x > 0:
        return gamma(x)
    if _is_near_nonpos_int(x):
        return float('inf')
    n = math.ceil(-x)
    x_p = x + n
    if x_p <= 0:
        return float('inf')
    g_p = gamma(x_p)
    prod = 1.0
    for k in range(n):
        prod *= (x + k)
    if prod == 0:
        return float('inf')
    return g_p / prod

def _series_2f1(a, b, c, z, max_terms=10000, tol=1e-15):
    if z == 0.0: return 1.0
    az = abs(z)
    if az >= 1.0: return float('nan')
    log_az = math.log(az)
    
    S = 1.0
    log_t = 0.0
    sign_t = 1.0
    converged = False
    
    for n in range(1, max_terms):
        try:
            log_poch_a = gammaln(a + n) - gammaln(a)
            log_poch_b = gammaln(b + n) - gammaln(b)
            log_poch_c = gammaln(c + n) - gammaln(c)
        except Exception:
            break
        
        log_t = log_poch_a + log_poch_b - log_poch_c - gammaln(n + 1.0) + n * log_az
        
        if log_t > 700:
            return float('inf')
        if log_t < -750:
            converged = True
            break
            
        t = math.exp(log_t)
        if (n % 2) != 0 and z < 0:
            t = -t
            
        S += t
        if abs(t) < tol * abs(S):
            converged = True
            break
            
    if not converged:
        return float('nan')
    return S

def _eval_2f1_safe(a, b, c, z):
    if abs(a) < 1e-12 or abs(b) < 1e-12:
        return 1.0
    if _is_near_nonpos_int(c):
        return None
    if abs(z) < 1.0:
        val = _safe_scipy_hyp2f1(a, b, c, z)
        if val is not None:
            return val
        return _series_2f1(a, b, c, z)
    return _safe_scipy_hyp2f1(a, b, c, z)

def _transform_1_z(a, b, c, z):
    z1 = z / (z - 1.0)
    if abs(z1) >= 1.0:
        return None
    
    cb = c - b
    ca = c - a
    has1 = not _is_near_nonpos_int(cb)
    has2 = not _is_near_nonpos_int(ca)
    
    if has1 and has2:
        if abs(cb) < abs(ca):
            val = _eval_2f1_safe(a, cb, c, z1)
            if val is not None:
                return (1.0 - z)**(-a) * val
        else:
            val = _eval_2f1_safe(ca, b, c, z1)
            if val is not None:
                return (1.0 - z)**(-b) * val
    elif has1:
        val = _eval_2f1_safe(a, cb, c, z1)
        if val is not None:
            return (1.0 - z)**(-a) * val
    elif has2:
        val = _eval_2f1_safe(ca, b, c, z1)
        if val is not None:
            return (1.0 - z)**(-b) * val
    return None

def _transform_1_z_inv(a, b, c, z):
    z1 = 1.0 - z
    if abs(z1) >= 1.0:
        return None
    
    ca = c - a
    cb = c - b
    cab = c - a - b
    has1 = not _is_near_nonpos_int(a)
    has2 = not _is_near_nonpos_int(b)
    
    if has1 and has2:
        if abs(a) < abs(b):
            val = _eval_2f1_safe(cab, a, ca, z1)
            if val is not None:
                return z**(-a) * val
        else:
            val = _eval_2f1_safe(cab, b, cb, z1)
            if val is not None:
                return z**(-b) * val
    elif has1:
        val = _eval_2f1_safe(cab, a, ca, z1)
        if val is not None:
            return z**(-a) * val
    elif has2:
        val = _eval_2f1_safe(cab, b, cb, z1)
        if val is not None:
            return z**(-b) * val
    return None

def _transform_z_inv(a, b, c, z):
    z_inv = 1.0 / z
    if abs(z_inv) >= 1.0:
        return None
    
    d = a - b
    has1 = not _is_near_nonpos_int(d)
    has2 = not _is_near_nonpos_int(-d)
    
    term1_val, term2_val = 0.0, 0.0
    has_t1, has_t2 = False, False
    
    if has1:
        try:
            g1 = _gamma_val(c)
            g2 = _gamma_val(b - a)
            g3 = _gamma_val(b)
            g4 = _gamma_val(c - a)
            if math.isfinite(g1) and math.isfinite(g2) and math.isfinite(g3) and math.isfinite(g4) and g3 != 0 and g4 != 0:
                pref1 = g1 * g2 / (g3 * g4)
                h1 = _eval_2f1_safe(a, a - c + 1.0, a - b + 1.0, z_inv)
                if h1 is not None:
                    term1_val = pref1 * (-z)**(-a) * h1
                    has_t1 = True
        except Exception:
            pass

    if has2:
        try:
            g1 = _gamma_val(c)
            g2 = _gamma_val(a - b)
            g3 = _gamma_val(a)
            g4 = _gamma_val(c - b)
            if math.isfinite(g1) and math.isfinite(g2) and math.isfinite(g3) and math.isfinite(g4) and g3 != 0 and g4 != 0:
                pref2 = g1 * g2 / (g3 * g4)
                h2 = _eval_2f1_safe(b, b - c + 1.0, b - a + 1.0, z_inv)
                if h2 is not None:
                    term2_val = pref2 * (-z)**(-b) * h2
                    has_t2 = True
        except Exception:
            pass

    if has_t1 and has_t2:
        res = term1_val + term2_val
        if math.isfinite(res) and res != 0.0:
            return res
    elif has_t1:
        if math.isfinite(term1_val) and term1_val != 0.0:
            return float(term1_val)
    elif has_t2:
        if math.isfinite(term2_val) and term2_val != 0.0:
            return float(term2_val)
    return None

def hyp2f1(a, b, c, z):
    if abs(a) > abs(b):
        a, b = b, a

    if z == 0.0:
        return 1.0

    if abs(z) < 1.0:
        val = _safe_scipy_hyp2f1(a, b, c, z)
        if val is not None:
            return val
        val = _series_2f1(a, b, c, z)
        if math.isfinite(val):
            return val

    if z < 0.0:
        z1 = z / (z - 1.0)
        if abs(z1) < 1.0:
            res = _transform_1_z(a, b, c, z)
            if res is not None and math.isfinite(res) and res != 0.0:
                return float(res)
        
        if abs(z) > 1.0:
            res = _transform_1_z_inv(a, b, c, z)
            if res is not None and math.isfinite(res) and res != 0.0:
                return float(res)
            
            res = _transform_z_inv(a, b, c, z)
            if res is not None and math.isfinite(res) and res != 0.0:
                return float(res)

    val = _safe_scipy_hyp2f1(a, b, c, z)
    if val is not None:
        return val

    return 0.0
