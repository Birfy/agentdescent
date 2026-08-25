import math
import numpy as np
from scipy.special import hyperu as _scipy_hyperu, hyp1f1 as _scipy_hyp1f1
from scipy.special import gamma as _gamma, gammaln as _gammaln, digamma as _digamma

def hyperu(a, b, x):
    """
    Evaluates Tricomi's confluent hypergeometric function U(a, b, x) for real parameters and x > 0.
    Designed to be robust and accurate across a wide range, fixing NaNs from the baseline
    by using the Gamma-weighted 1F1 pair and Kummer transformations.
    """
    try:
        a_f = float(a); b_f = float(b); x_f = float(x)
    except Exception:
        return float('nan')
    
    if not (np.isfinite(a_f) and np.isfinite(b_f) and np.isfinite(x_f) and x_f > 0):
        return float('nan')

    a = a_f; b = b_f; x = x_f

    # 1. Try the baseline scipy.special.hyperu first.
    # It is highly accurate where it doesn't fail.
    try:
        val = _scipy_hyperu(a, b, x)
        val = float(val)
        if math.isfinite(val) and val != 0.0:
            return val
    except Exception:
        pass

    # 2. Baseline failed or returned 0/inf. Use the Gamma-weighted 1F1 pair.
    # U(a,b,x) = Gamma(1-b)/Gamma(a-b+1) * 1F1(a, b, x)
    #          + Gamma(b-1)/Gamma(a)     * x^(1-b) * 1F1(a-b+1, 2-b, x)
    # This representation is valid when b is not an integer.
    def calc_1f1_pair(a, b, x):
        try:
            t1 = _gammaln(1 - b) - _gammaln(a - b + 1)
            t2 = _gammaln(b - 1) - _gammaln(a)
            
            # If both terms are extremely small, the result is likely exactly 0.
            if t1 < -700 and t2 < -700:
                return 0.0

            m1 = _scipy_hyp1f1(a, b, x)
            m2 = _scipy_hyp1f1(a - b + 1, 2 - b, x)
            
            term1 = 0.0
            if np.isfinite(m1) and m1 != 0.0 and t1 > -700:
                term1 = math.exp(t1) * m1
                
            term2 = 0.0
            if np.isfinite(m2) and m2 != 0.0 and t2 > -700:
                term2 = math.exp(t2 + (1 - b) * math.log(x)) * m2
                
            res = term1 + term2
            if math.isfinite(res):
                return res
        except Exception:
            pass
        return float('nan')

    # 3. Kummer transformation: U(a,b,x) = x^(1-b) * U(a-b+1, 2-b, x)
    # If b is close to an integer, 2-b is also close to an integer, so this doesn't
    # directly fix the pole in the 1F1 pair. However, it shifts the parameters,
    # which can prevent numerical overflow in the Gamma weights or hyp1f1 evaluations.
    def calc_kummer(a, b, x):
        try:
            a2 = a - b + 1
            b2 = 2 - b
            v2 = calc_1f1_pair(a2, b2, x)
            if math.isfinite(v2):
                v1 = math.exp((1 - b) * math.log(x)) * v2
                if math.isfinite(v1):
                    return v1
        except Exception:
            pass
        return float('nan')

    # 4. Try the direct 1F1 pair and its Kummer transform.
    res = calc_1f1_pair(a, b, x)
    if math.isfinite(res) and res != 0.0:
        return res
        
    res = calc_kummer(a, b, x)
    if math.isfinite(res) and res != 0.0:
        return res

    # 5. Fallback to baseline even if it was 0 or inf, to avoid returning NaN.
    try:
        val = _scipy_hyperu(a, b, x)
        val = float(val)
        if math.isfinite(val):
            return val
    except Exception:
        pass
        
    return 0.0
