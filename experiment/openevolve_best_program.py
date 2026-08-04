"""Adaptive simplex (Nelder-Mead) local search with multi-start from stratified samples."""

import math

def search_algorithm(objective, budget, rng, bounds):
    """Return the best (x, y) found within a strict objective-call budget."""
    low, high = bounds
    width = high - low
    calls_made = 0

    def clip(v):
        if v < low: return low
        if v > high: return high
        return v

    def obj(x, y):
        nonlocal calls_made
        if calls_made >= budget:
            return float('inf')
        calls_made += 1
        return objective(x, y)

    # Phase 1: Stratified sampling for broad coverage
    n_strata = 7
    stratum_size = width / n_strata
    candidates = []

    for i in range(n_strata):
        for j in range(n_strata):
            if calls_made >= budget: break
            x = clip(low + (i + rng.random()) * stratum_size)
            y = clip(low + (j + rng.random()) * stratum_size)
            val = obj(x, y)
            if val != float('inf'):
                candidates.append((val, x, y))
        if calls_made >= budget: break

    candidates.sort()
    if not candidates:
        return clip(low + width * 0.5), clip(low + width * 0.5)

    best_val, best_x, best_y = candidates[0]

    # Phase 2: Multi-start Nelder-Mead simplex local search
    n_starts = min(4, len(candidates))

    for k in range(n_starts):
        if calls_made >= budget - 4: break
        _, sx, sy = candidates[k]

        # Initialize simplex
        s = width * 0.15
        v0 = (sx, sy)
        v1 = (clip(sx + s), sy)
        v2 = (sx, clip(sy + s))

        f0 = obj(v0[0], v0[1])
        f1 = obj(v1[0], v1[1])
        f2 = obj(v2[0], v2[1])

        if f0 < best_val: best_val, best_x, best_y = f0, v0[0], v0[1]
        if f1 < best_val: best_val, best_x, best_y = f1, v1[0], v1[1]
        if f2 < best_val: best_val, best_x, best_y = f2, v2[0], v2[1]

        # Nelder-Mead parameters
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 60
        tol_s = 1e-5
        tol_f = 1e-8

        for _ in range(max_iter):
            if calls_made >= budget - 1: break

            # Sort vertices by value
            verts = [(f0, v0), (f1, v1), (f2, v2)]
            verts.sort()
            (f0, v0), (f1, v1), (f2, v2) = verts

            # Convergence check
            if abs(f2 - f0) < tol_f: break
            d01 = math.hypot(v1[0]-v0[0], v1[1]-v0[1])
            d02 = math.hypot(v2[0]-v0[0], v2[1]-v0[1])
            if max(d01, d02) < tol_s: break

            # Centroid of best two
            cx = (v0[0] + v1[0]) / 2.0
            cy = (v0[1] + v1[1]) / 2.0

            # Reflection
            rx = clip(cx + alpha * (cx - v2[0]))
            ry = clip(cy + alpha * (cy - v2[1]))
            fr = obj(rx, ry)
            if fr < best_val: best_val, best_x, best_y = fr, rx, ry

            if f0 <= fr < f1:
                v2, f2 = (rx, ry), fr
            elif fr < f0:
                # Expansion
                ex = clip(cx + gamma * (rx - cx))
                ey = clip(cy + gamma * (ry - cy))
                fe = obj(ex, ey)
                if fe < best_val: best_val, best_x, best_y = fe, ex, ey
                if fe < fr:
                    v2, f2 = (ex, ey), fe
                else:
                    v2, f2 = (rx, ry), fr
            else:
                # Contraction
                if fr < f2:
                    cxp = clip(cx + rho * (rx - cx))
                    cyp = clip(cy + rho * (ry - cy))
                    fc = obj(cxp, cyp)
                    if fc < best_val: best_val, best_x, best_y = fc, cxp, cyp
                    if fc <= fr:
                        v2, f2 = (cxp, cyp), fc
                    else:
                        # Shrink
                        v1 = (clip(v0[0] + sigma * (v1[0] - v0[0])), clip(v0[1] + sigma * (v1[1] - v0[1])))
                        f1 = obj(v1[0], v1[1])
                        if f1 < best_val: best_val, best_x, best_y = f1, v1[0], v1[1]
                        if calls_made >= budget: break
                        v2 = (clip(v0[0] + sigma * (v2[0] - v0[0])), clip(v0[1] + sigma * (v2[1] - v0[1])))
                        f2 = obj(v2[0], v2[1])
                        if f2 < best_val: best_val, best_x, best_y = f2, v2[0], v2[1]
                else:
                    cxp = clip(cx + rho * (v2[0] - cx))
                    cyp = clip(cy + rho * (v2[1] - cy))
                    fc = obj(cxp, cyp)
                    if fc < best_val: best_val, best_x, best_y = fc, cxp, cyp
                    if fc < f2:
                        v2, f2 = (cxp, cyp), fc
                    else:
                        v1 = (clip(v0[0] + sigma * (v1[0] - v0[0])), clip(v0[1] + sigma * (v1[1] - v0[1])))
                        f1 = obj(v1[0], v1[1])
                        if f1 < best_val: best_val, best_x, best_y = f1, v1[0], v1[1]
                        if calls_made >= budget: break
                        v2 = (clip(v0[0] + sigma * (v2[0] - v0[0])), clip(v0[1] + sigma * (v2[1] - v0[1])))
                        f2 = obj(v2[0], v2[1])
                        if f2 < best_val: best_val, best_x, best_y = f2, v2[0], v2[1]

    return best_x, best_y
