import math
import numpy as np
from scipy.special import pbdv, pbvv as _scipy_pbvv, hyperu, gammaln, hyp1f1, gamma, poch, digamma

def _log_factorial(n):
    return float(gammaln(n + 1))

def _eval_1f1_series(a, b, z):
    s = 1.0
    t = 1.0
    for k in range(1, 200):
        t *= (a + k - 1) * z / ((b + k - 1) * k)
        s_new = s + t
        if abs(s_new - s) <= 1e-16 * abs(s_new):
            break
        s = s_new
    return s

def _eval_1f1_asym_log(a, b, z):
    z2 = z / 2.0
    rho = math.sqrt(1.0 - b / z)
    a1 = z2 * rho
    a2 = z2 - a1
    p1 = a - a1
    p2 = a - a2
    q1 = (b / 2.0 - 1.0 / 4.0) * (1.0 - rho)
    q2 = (b / 2.0 - 1.0 / 4.0) * (1.0 + rho)
    log_pref = -z2 + (p1 - 0.5) * math.log(z) - gammaln(p1)
    log_t1 = p1 * math.log(a1) - a1 + gammaln(2 * p1 - 1) + q1 / a1
    log_t2 = p2 * math.log(a2) - a2 + gammaln(2 * p2 - 1) + q2 / a2
    if log_t1 > log_t2:
        log_M = log_pref + log_t1 + math.log1p(math.exp(log_t2 - log_t1))
    else:
        log_M = log_pref + log_t2 + math.log1p(math.exp(log_t1 - log_t2))
    return log_M

def _pbvv_neg_x(v, x):
    v = float(v)
    x = float(x)
    abs_x = abs(x)
    
    # Fallback to scipy for small v or x, where it is reliable
    if v < 9.0 or abs_x < 10.0:
        return float(_scipy_pbvv(v, x)[0])
        
    # Use asymptotic expansion for large |x|
    if abs_x > 20.0:
        z = abs_x * math.sqrt(2.0)
        v2 = v / 2.0
        phi = z * z / 4.0 - v2 * math.pi / 2.0 - math.pi / 8.0
        N = int(math.floor(v + 0.5))
        alpha = v - N
        s = 0.0
        for k in range(1, 15):
            a1k = 1.0
            for j in range(1, k):
                a1k *= (4.0 * v * v - (2.0 * j - 1.0) ** 2) / (32.0 * j)
            a2k = 1.0
            for j in range(1, k + 1):
                a2k *= (4.0 * v * v - (2.0 * j - 1.0) ** 2) / (32.0 * j)
            s += a1k * math.cos((2.0 * k - 1.0) * phi) / z ** (2.0 * k - 1.0)
            s += a2k * math.cos((2.0 * k) * phi) / z ** (2.0 * k)
        log_D = -z * z / 4.0 + v2 * math.log(z) - 0.5 * math.log(2.0 * math.pi)
        D_abs = math.exp(log_D) * (math.cos(phi) + s)
        
        if N % 2 == 0:
            log_K = gammaln(-v)
            A = math.cos(alpha * math.pi)
            B = math.sin(alpha * math.pi)
            log_C = log_K - math.log(math.pi)
            V = math.exp(log_C) * (A * D_abs + B * math.exp(-log_D) * math.sqrt(2.0 * math.pi) * math.exp(gammaln(v + 1.0)))
        else:
            log_K = gammaln(-v)
            A = -math.sin(alpha * math.pi)
            B = math.cos(alpha * math.pi)
            log_C = log_K - math.log(math.pi)
            V = math.exp(log_C) * (A * D_abs + B * math.exp(-log_D) * math.sqrt(2.0 * math.pi) * math.exp(gammaln(v + 1.0)))
        
        if x < 0:
            return float(V)
        else:
            return float(-V)
            
    # Use confluent hypergeometric formulation for intermediate x
    # V_v(-|x|) = G(-v) * [ D_v(-|x|) + cos(pi v) D_v(|x|) ]
    # D_v(|x|) = 2^(v/2) * exp(-x^2/4) * U(-v/2, 1/2, x^2/2)
    # D_v(-|x|) = 2^(v/2) * sqrt(pi) * exp(-x^2/4) * [ 1F1 / G((1-v)/2) + sqrt(2)*|x|*1F1 / G(-v/2) ]
    z_h = x * x / 2.0
    a_u = -v / 2.0
    b_u = 0.5
    
    # Calculate D_v(|x|) via hyperu
    log_pref_u = (v / 2.0) * math.log(2.0) - x * x / 4.0
    if z_h > 25.0 and a_u < 0:
        # Use asymptotic expansion for U to prevent overflow/loss of precision
        log_U = _eval_1f1_asym_log(a_u, b_u, z_h)
        log_D_pos = log_pref_u + log_U
        D_pos = math.exp(log_D_pos)
    else:
        # Direct evaluation
        U_val = float(hyperu(a_u, b_u, z_h))
        D_pos = math.exp(log_pref_u) * U_val
        
    # Calculate D_v(-|x|) via 1F1
    a1 = -v / 2.0
    b1 = 0.5
    a2 = (1.0 - v) / 2.0
    b2 = 1.5
    
    if z_h < 20.0:
        m1 = _eval_1f1_series(a1, b1, z_h)
        m2 = _eval_1f1_series(a2, b2, z_h)
    else:
        m1 = float(hyp1f1(a1, b1, z_h))
        m2 = float(hyp1f1(a2, b2, z_h))
        
    g1 = float(gamma((1.0 - v) / 2.0))
    g2 = float(gamma(-v / 2.0))
    
    val_1f1 = m1 / g1 + math.sqrt(2.0) * abs_x * m2 / g2
    log_pref_1f1 = (v / 2.0) * math.log(2.0) + 0.5 * math.log(math.pi) - x * x / 4.0
    D_neg = math.exp(log_pref_1f1) * val_1f1
    
    # Combine to get V_v(-|x|)
    G_neg_v = float(gamma(-v))
    cos_pv = math.cos(math.pi * v)
    
    V_val = (G_neg_v / math.pi) * (D_neg + cos_pv * D_pos)
    
    if x < 0:
        return float(V_val)
    else:
        return float(-V_val)

def pbvv(v, x):
    v = float(v)
    x = float(x)
    
    # The problematic region for scipy is large positive v and negative x
    if v >= 9.0 and x < 0.0:
        return _pbvv_neg_x(v, x)
        
    # For x >= 0, or v < 9, scipy is reliable
    return float(_scipy_pbvv(v, x)[0])
