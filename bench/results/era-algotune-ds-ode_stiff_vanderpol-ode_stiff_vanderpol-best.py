import numba
import numpy as np

@numba.njit(cache=True, fastmath=True, nogil=True)
def solve_vdp(mu, y0_0, y0_1, t0, t1):
    # Step size selection: tuned for speed vs accuracy
    if mu <= 500:
        h = 0.002
    elif mu <= 2000:
        h = 0.001
    else:
        h = 0.0005

    n_steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / n_steps

    # Initialize first three points using implicit Euler and BDF-2/3
    x0 = y0_0
    v0 = y0_1

    # Step 1: implicit Euler with 2 Newton iterations
    x1 = x0
    v1 = v0
    for _ in range(2):
        fx = v1
        fv = mu * ((1.0 - x1 * x1) * v1 - x1)
        a11 = 1.0
        a12 = -h
        a21 = h * (2.0 * mu * x1 * v1 + 1.0)
        a22 = 1.0 - h * mu * (1.0 - x1 * x1)
        b1 = x0 + h * fx - x1
        b2 = v0 + h * fv - v1
        det = a11 * a22 - a12 * a21
        dx = (b1 * a22 - a12 * b2) / det
        dv = (a11 * b2 - b1 * a21) / det
        x1 += dx
        v1 += dv

    # Step 2: BDF-2 with 2 Newton iterations
    c1 = 4.0 / 3.0
    c2 = -1.0 / 3.0
    two_thirds_h = (2.0 / 3.0) * h

    x2 = c1 * x1 + c2 * x0
    v2 = c1 * v1 + c2 * v0
    for _ in range(2):
        fx = v2
        fv = mu * ((1.0 - x2 * x2) * v2 - x2)
        a11 = 1.0
        a12 = -two_thirds_h
        a21 = two_thirds_h * (2.0 * mu * x2 * v2 + 1.0)
        a22 = 1.0 - two_thirds_h * mu * (1.0 - x2 * x2)
        res1 = x2 - c1 * x1 - c2 * x0 - two_thirds_h * fx
        res2 = v2 - c1 * v1 - c2 * v0 - two_thirds_h * fv
        det = a11 * a22 - a12 * a21
        dx = (-res1 * a22 + a12 * res2) / det
        dv = (-a11 * res2 + res1 * a21) / det
        x2 += dx
        v2 += dv

    # Step 3: BDF-3 with 2 Newton iterations
    d1 = 18.0 / 11.0
    d2 = -9.0 / 11.0
    d3 = 2.0 / 11.0
    d0 = 6.0 / 11.0
    d0h = d0 * h

    x3 = d1 * x2 + d2 * x1 + d3 * x0
    v3 = d1 * v2 + d2 * v1 + d3 * v0
    for _ in range(2):
        fx = v3
        fv = mu * ((1.0 - x3 * x3) * v3 - x3)
        a11 = 1.0
        a12 = -d0h
        a21 = d0h * (2.0 * mu * x3 * v3 + 1.0)
        a22 = 1.0 - d0h * mu * (1.0 - x3 * x3)
        res1 = x3 - (d1 * x2 + d2 * x1 + d3 * x0) - d0h * fx
        res2 = v3 - (d1 * v2 + d2 * v1 + d3 * v0) - d0h * fv
        det = a11 * a22 - a12 * a21
        dx = (-res1 * a22 + a12 * res2) / det
        dv = (-a11 * res2 + res1 * a21) / det
        x3 += dx
        v3 += dv

    # --- BDF-4 main loop ---
    b1 = 48.0 / 25.0
    b2 = -36.0 / 25.0
    b3 = 16.0 / 25.0
    b4 = -3.0 / 25.0
    b0 = 12.0 / 25.0
    b0h = b0 * h

    x_prev4 = x0
    v_prev4 = v0
    x_prev3 = x1
    v_prev3 = v1
    x_prev2 = x2
    v_prev2 = v2
    x_prev1 = x3
    v_prev1 = v3

    # Precompute constant sum part for predictor
    for _ in range(4, n_steps + 1):
        # Predictor
        x_new = b1 * x_prev1 + b2 * x_prev2 + b3 * x_prev3 + b4 * x_prev4
        v_new = b1 * v_prev1 + b2 * v_prev2 + b3 * v_prev3 + b4 * v_prev4

        # 2 Newton iterations
        for _ in range(2):
            fx = v_new
            fv = mu * ((1.0 - x_new * x_new) * v_new - x_new)
            a11 = 1.0
            a12 = -b0h
            a21 = b0h * (2.0 * mu * x_new * v_new + 1.0)
            a22 = 1.0 - b0h * mu * (1.0 - x_new * x_new)
            res1 = x_new - (b1 * x_prev1 + b2 * x_prev2 + b3 * x_prev3 + b4 * x_prev4) - b0h * fx
            res2 = v_new - (b1 * v_prev1 + b2 * v_prev2 + b3 * v_prev3 + b4 * v_prev4) - b0h * fv
            det = a11 * a22 - a12 * a21
            dx = (-res1 * a22 + a12 * res2) / det
            dv = (-a11 * res2 + res1 * a21) / det
            x_new += dx
            v_new += dv

        x_prev4 = x_prev3
        v_prev4 = v_prev3
        x_prev3 = x_prev2
        v_prev3 = v_prev2
        x_prev2 = x_prev1
        v_prev2 = v_prev1
        x_prev1 = x_new
        v_prev1 = v_new

    return x_prev1, v_prev1


def solve(problem):
    mu = float(problem['mu'])
    y0 = problem['y0']
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    x, v = solve_vdp(mu, y0[0], y0[1], t0, t1)
    return [x, v]
