import numpy as np
import itertools
import time
import math

def discover(x, y, spec):
    start_time = time.time()
    time_limit = 3.6

    names = list(spec["input_vars"])
    n_inputs = len(names)
    n_samples = x.shape[0]

    best_expr = repr(float(np.mean(y)))
    best_nmse = np.inf

    y_var = np.var(y)
    if y_var < 1e-14:
        y_var = 1.0

    if n_samples < 10:
        return best_expr

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    n_val = max(5, n_samples // 3)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    x_fit, y_fit = x[fit_idx], y[fit_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    y_var_val = np.var(y_val) + 1e-12

    def update(expr, nmse):
        nonlocal best_expr, best_nmse
        if nmse < best_nmse:
            best_nmse = nmse
            best_expr = expr

    def try_terms(terms):
        nonlocal best_expr, best_nmse
        if time.time() - start_time > time_limit:
            return
        if not terms:
            return
        
        funcs = [t[0] for t in terms]
        texts = [t[1] for t in terms]
        
        try:
            B_fit = np.column_stack([f(x_fit) for f in funcs])
            B_val = np.column_stack([f(x_val) for f in funcs])
        except Exception:
            return

        if not np.all(np.isfinite(B_fit)) or not np.all(np.isfinite(B_val)):
            return

        p = B_fit.shape[1]
        if p == 0:
            return

        scale = np.maximum(np.max(np.abs(B_fit), axis=0), 1e-12)
        B_fit_s = B_fit / scale
        B_val_s = B_val / scale

        try:
            c, *_ = np.linalg.lstsq(B_fit_s, y_fit, rcond=None)
        except Exception:
            return

        if not np.all(np.isfinite(c)):
            return

        y_val_pred = B_val_s @ c
        nmse = np.mean((y_val_pred - y_val)**2) / y_var_val
        if not np.isfinite(nmse):
            return

        parts = []
        for i in range(p):
            if abs(c[i]) > 1e-10:
                coeff = c[i] / scale[i]
                parts.append(f"({coeff:.17g})*{texts[i]}")
        
        if not parts:
            return

        expr = " + ".join(parts)
        try:
            y_pred = spec["evaluate"](expr, x_val)
            if y_pred is not None:
                y_pred = np.asarray(y_pred, dtype=np.float64)
                if y_pred.shape[0] == x_val.shape[0] and np.all(np.isfinite(y_pred)):
                    nmse_eval = np.mean((y_pred - y_val)**2) / y_var_val
                    if np.isfinite(nmse_eval):
                        update(expr, nmse_eval)
        except Exception:
            pass

    const_term = (lambda x: np.ones(x.shape[0]), "1")
    try_terms([const_term])
    
    single_terms = [(lambda x, i=i: x[:, i], names[i]) for i in range(n_inputs)]
    try_terms([const_term] + single_terms)

    all_terms = [const_term]
    for i in range(n_inputs):
        def f(x, i=i): return x[:, i]
        all_terms.append((f, names[i]))
        def f2(x, i=i): return x[:, i]**2
        all_terms.append((f2, f"{names[i]}**2"))
        def f3(x, i=i): return x[:, i]**3
        all_terms.append((f3, f"{names[i]}**3"))
        def finv(x, i=i): return 1.0 / (x[:, i] + 1e-12)
        all_terms.append((finv, f"1/{names[i]}"))
        def finv2(x, i=i): return 1.0 / (x[:, i]**2 + 1e-12)
        all_terms.append((finv2, f"1/{names[i]}**2"))
        def fsqr(x, i=i): return np.sqrt(np.abs(x[:, i]))
        all_terms.append((fsqr, f"sqrt(Abs({names[i]}))"))
        def fsin(x, i=i): return np.sin(x[:, i])
        all_terms.append((fsin, f"sin({names[i]})"))
        def fcos(x, i=i): return np.cos(x[:, i])
        all_terms.append((fcos, f"cos({names[i]})"))
        def fexp(x, i=i): return np.exp(x[:, i])
        all_terms.append((fexp, f"exp({names[i]})"))
        def fexn(x, i=i): return np.exp(-x[:, i])
        all_terms.append((fexn, f"exp(-{names[i]})"))
        def flog(x, i=i): return np.log(np.abs(x[:, i]) + 1e-12)
        all_terms.append((flog, f"log(Abs({names[i]}))"))

    for i in range(n_inputs):
        for j in range(i, n_inputs):
            def fm(x, i=i, j=j): return x[:, i] * x[:, j]
            all_terms.append((fm, f"{names[i]}*{names[j]}"))

    try_terms(all_terms)

    if time.time() - start_time < time_limit:
        ratios = []
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                def fr(x, i=i, j=j): return x[:, i] / (x[:, j] + 1e-12)
                ratios.append((fr, f"{names[i]}/{names[j]}"))
        try_terms([const_term] + single_terms + ratios)

    if time.time() - start_time < time_limit:
        ratios3 = []
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                for k in range(n_inputs):
                    if k == i or k == j: continue
                    def fr3(x, i=i, j=j, k=k): return x[:, i] * x[:, j] / (x[:, k] + 1e-12)
                    ratios3.append((fr3, f"{names[i]}*{names[j]}/{names[k]}"))
        try_terms([const_term] + single_terms + ratios3)

    if time.time() - start_time < time_limit:
        ratios4 = []
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                for k in range(n_inputs):
                    if k == i or k == j: continue
                    for l in range(n_inputs):
                        if l == i or l == j or l == k: continue
                        def fr4(x, i=i, j=j, k=k, l=l): return x[:, i] * x[:, j] / (x[:, k] * x[:, l] + 1e-12)
                        ratios4.append((fr4, f"{names[i]}*{names[j]}/({names[k]}*{names[l]})"))
        try_terms([const_term] + single_terms + ratios4)

    if time.time() - start_time < time_limit:
        ratios5 = []
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                for k in range(n_inputs):
                    if k == i or k == j: continue
                    for l in range(n_inputs):
                        if l == i or l == j or l == k: continue
                        for m in range(n_inputs):
                            if m == i or m == j or m == k or m == l: continue
                            def fr5(x, i=i, j=j, k=k, l=l, m=m): return x[:, i] * x[:, j] * x[:, k] / (x[:, l] * x[:, m] + 1e-12)
                            ratios5.append((fr5, f"{names[i]}*{names[j]}*{names[k]}/({names[l]}*{names[m]})"))
        try_terms([const_term] + single_terms + ratios5)

    if time.time() - start_time < time_limit:
        ratios6 = []
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                for k in range(n_inputs):
                    if k == i or k == j: continue
                    for l in range(n_inputs):
                        if l == i or l == j or l == k: continue
                        for m in range(n_inputs):
                            if m == i or m == j or m == k or m == l: continue
                            for o in range(n_inputs):
                                if o == i or o == j or o == k or o == l or o == m: continue
                                def fr6(x, i=i, j=j, k=k, l=l, m=m, o=o): return x[:, i] * x[:, j] * x[:, k] / (x[:, l] * x[:, m] * x[:, o] + 1e-12)
                                ratios6.append((fr6, f"{names[i]}*{names[j]}*{names[k]}/({names[l]}*{names[m]}*{names[o]})"))
        try_terms([const_term] + single_terms + ratios6)

    if time.time() - start_time < time_limit:
        ratios7 = []
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                for k in range(n_inputs):
                    if k == i or k == j: continue
                    for l in range(n_inputs):
                        if l == i or l == j or l == k: continue
                        for m in range(n_inputs):
                            if m == i or m == j or m == k or m == l: continue
                            for o in range(n_inputs):
                                if o == i or o == j or o == k or o == l or o == m: continue
                                for p in range(n_inputs):
                                    if p == i or p == j or p == k or p == l or p == m or p == o: continue
                                    def fr7(x, i=i, j=j, k=k, l=l, m=m, o=o, p=p): return x[:, i] * x[:, j] * x[:, k] * x[:, l] / (x[:, m] * x[:, o] * x[:, p] + 1e-12)
                                    ratios7.append((fr7, f"{names[i]}*{names[j]}*{names[k]}*{names[l]}/({names[m]}*{names[o]}*{names[p]})"))
        try_terms([const_term] + single_terms + ratios7)

    if best_nmse < 1e-10:
        return best_expr

    if time.time() - start_time < time_limit:
        base_terms = [const_term] + single_terms
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                def fr(x, i=i, j=j): return x[:, i] / (x[:, j] + 1e-12)
                r_term = (fr, f"{names[i]}/{names[j]}")
                try_terms(base_terms + [r_term])

    if time.time() - start_time < time_limit:
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                def fr(x, i=i, j=j): return x[:, i] / (x[:, j] + 1e-12)
                for k in range(n_inputs):
                    def fk(x, k=k): return x[:, k]
                    try_terms([const_term, (fr, f"{names[i]}/{names[j]}"), (fk, names[k])])

    if time.time() - start_time < time_limit:
        for i in range(n_inputs):
            for j in range(n_inputs):
                if i == j: continue
                def fr(x, i=i, j=j): return x[:, i] / (x[:, j] + 1e-12)
                for k in range(n_inputs):
                    for l in range(n_inputs):
                        if k == l: continue
                        def fr2(x, k=k, l=l): return x[:, k] / (x[:, l] + 1e-12)
                        try_terms([const_term, (fr, f"{names[i]}/{names[j]}"), (fr2, f"{names[k]}/{names[l]}")])

    if time.time() - start_time < time_limit:
        for i in range(n_inputs):
            for j in range(i + 1, n_inputs):
                def fp(x, i=i, j=j): return x[:, i] * x[:, j]
                for k in range(n_inputs):
                    for l in range(n_inputs):
                        if k == l: continue
                        def fr(x, k=k, l=l): return x[:, k] / (x[:, l] + 1e-12)
                        try_terms([const_term, (fp, f"{names[i]}*{names[j]}"), (fr, f"{names[k]}/{names[l]}")])

    return best_expr
