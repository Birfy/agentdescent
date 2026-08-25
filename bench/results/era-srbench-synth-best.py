import numpy as np
import math
import time
import itertools
import functools

def _add_term(terms, vals, txt):
    if np.all(np.isfinite(vals)):
        terms.append((vals, txt))

def _build_base_terms(x, names):
    n, m = x.shape
    terms = []
    _add_term(terms, np.ones(n), "1")
    
    cols = {}
    for i, name in enumerate(names):
        v = x[:, i]
        cols[name] = v
        _add_term(terms, v, name)
        _add_term(terms, v * v, f"{name}**2")
        _add_term(terms, v * v * v, f"{name}**3")
        _add_term(terms, v * v * v * v, f"{name}**4")
        
        _add_term(terms, np.sin(v), f"sin({name})")
        _add_term(terms, np.cos(v), f"cos({name})")
        _add_term(terms, np.tanh(v), f"tanh({name})")
        
        if np.all(v > 1e-9):
            _add_term(terms, np.log(v), f"log({name})")
            _add_term(terms, np.sqrt(v), f"sqrt({name})")
        if np.min(np.abs(v)) > 1e-6:
            _add_term(terms, 1.0 / v, f"1/({name})")
        if np.max(np.abs(v)) < 30.0:
            _add_term(terms, np.exp(v), f"exp({name})")
            _add_term(terms, np.exp(-v), f"exp(-({name}))")
        elif np.max(np.abs(v)) < 700.0:
            _add_term(terms, np.exp(-v), f"exp(-({name}))")
            
    if m >= 2:
        for i in range(m):
            for j in range(i + 1, m):
                v1, v2 = x[:, i], x[:, j]
                _add_term(terms, v1 * v2, f"{names[i]}*{names[j]}")
                _add_term(terms, v1 * v1 * v2, f"{names[i]}**2*{names[j]}")
                _add_term(terms, v1 * v2 * v2, f"{names[i]}*{names[j]}**2")
                
                _add_term(terms, np.sin(v1) * v2, f"sin({names[i]})*{names[j]}")
                _add_term(terms, v1 * np.sin(v2), f"{names[i]}*sin({names[j]})")
                _add_term(terms, np.cos(v1) * v2, f"cos({names[i]})*{names[j]}")
                _add_term(terms, v1 * np.cos(v2), f"{names[i]}*cos({names[j]})")
                
    if m >= 3:
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    _add_term(terms, x[:, i] * x[:, j] * x[:, k], f"{names[i]}*{names[j]}*{names[k]}")
                    
    return terms

def _fit_and_score(A, y, y_scale):
    if A.shape[1] == 0:
        return np.inf, None
    try:
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return np.inf, None
    res = y - A @ coeffs
    err = np.mean(res**2) / (y_scale + 1e-12)
    return err, coeffs

def _stlsq(A, y, y_scale, max_sweeps=12, max_terms=None):
    try:
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return np.inf, None

    for _ in range(max_sweeps):
        active = np.abs(coeffs) > 1e-6 * np.sqrt(y_scale)
        if not active.any():
            active = np.zeros(A.shape[1], dtype=bool)
            active[np.argmax(np.abs(coeffs))] = True
        if max_terms is not None and np.sum(active) > max_terms:
            idx = np.argsort(-np.abs(coeffs))
            active = np.zeros(A.shape[1], dtype=bool)
            active[idx[:max_terms]] = True
        if not active.any():
            break
        try:
            sub_coeffs, *_ = np.linalg.lstsq(A[:, active], y, rcond=None)
        except Exception:
            break
        new_coeffs = np.zeros(A.shape[1])
        new_coeffs[active] = sub_coeffs
        if np.allclose(new_coeffs, coeffs):
            coeffs = new_coeffs
            break
        coeffs = new_coeffs
        
    res = y - A @ coeffs
    err = np.mean(res**2) / (y_scale + 1e-12)
    return err, coeffs

def discover(x, y, spec):
    start = time.time()
    names = list(spec["input_vars"])
    n, m = x.shape
    evaluate = spec["evaluate"]
    
    y_mean = float(np.mean(y))
    y_var = float(np.var(y))
    y_scale = y_var if y_var > 1e-12 else 1.0
    
    best_score = -0.0
    best_eq = repr(y_mean)
    
    try:
        y_pred = evaluate(best_eq, x)
        if np.all(np.isfinite(y_pred)):
            res = y - y_pred
            err = np.mean(res**2) / y_scale
            best_score = min(12.0, -math.log10(err + 1e-300)) if err > 0 else 12.0
    except Exception:
        pass
        
    if n < 8:
        return best_eq
        
    terms = _build_base_terms(x, names)
    if not terms:
        return best_eq
        
    A = np.column_stack([t[0] for t in terms])
    labels = [t[1] for t in terms]
    
    scale = np.maximum(np.abs(A).max(axis=0), 1e-12)
    A_norm = A / scale
    y_norm = y
    
    def _try_eq(coeffs, idx):
        nonlocal best_score, best_eq
        nz = np.abs(coeffs) > 1e-12
        if not np.any(nz):
            return
        parts = [f"({float(coeffs[i]/scale[idx[i]]):.12g})*{labels[idx[i]]}" for i in range(len(idx)) if nz[i]]
        if not parts:
            return
        eq = " + ".join(parts)
        try:
            y_pred = evaluate(eq, x)
            if np.all(np.isfinite(y_pred)):
                res = y - y_pred
                err = np.mean(res**2) / y_scale
                score = min(12.0, -math.log10(err + 1e-300)) if err > 0 else 12.0
                if score > best_score:
                    best_score = score
                    best_eq = eq
        except Exception:
            pass

    for max_t in (5, 8, 12, 18, 30):
        if time.time() - start > 1.5:
            break
        err, coeffs = _stlsq(A_norm, y_norm, y_scale, max_terms=max_t)
        if coeffs is not None:
            _try_eq(coeffs, list(range(len(terms))))
            
    indices = list(range(len(terms)))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    
    tested = set()
    for k in range(2, 7):
        if time.time() - start > 3.5:
            break
        for _ in range(40):
            if time.time() - start > 3.5:
                break
            idx = tuple(sorted(rng.choice(indices, k, replace=False).tolist()))
            if idx in tested:
                continue
            tested.add(idx)
            
            A_sub = A_norm[:, list(idx)]
            err, coeffs = _fit_and_score(A_sub, y_norm, y_scale)
            if coeffs is not None:
                _try_eq(coeffs, list(idx))
                
    return best_eq
