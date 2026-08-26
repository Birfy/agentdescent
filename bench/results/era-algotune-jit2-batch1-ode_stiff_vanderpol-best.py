import numpy as np
from numba import njit


@njit(cache=True)
def _vdp_integrate(mu, x0, v0, t0, t1):
    x = x0
    v = v0
    t = t0

    rtol = 1e-8
    atol = 1e-9
    max_steps = 10000000
    h_abs = 1e-6

    C1 = 1.0 / 6.0
    C2 = 1.0 / 30.0

    for _ in range(max_steps):
        if t >= t1:
            return x, v

        dist = t1 - t
        if h_abs > dist:
            h_abs = dist

        h = h_abs
        h2 = 0.5 * h

        f1x = v
        f1v = mu * ((1.0 - x * x) * v - x)

        y2x = x + h2 * f1x
        y2v = v + h2 * f1v

        f2x = y2v
        f2v = mu * ((1.0 - y2x * y2x) * y2v - y2x)

        y3x = x + h2 * f2x
        y3v = v + h2 * f2v

        f3x = y3v
        f3v = mu * ((1.0 - y3x * y3x) * y3v - y3x)

        y4x = x + h * f3x
        y4v = v + h * f3v

        f4x = y4v
        f4v = mu * ((1.0 - y4x * y4x) * y4v - y4x)

        y5x = x + h * (f1x + 2.0 * f2x + 2.0 * f3x + f4x) * C1
        y5v = v + h * (f1v + 2.0 * f2v + 2.0 * f3v + f4v) * C1

        f5x = y5v
        f5v = mu * ((1.0 - y5x * y5x) * y5v - y5x)

        err_x = h * (-5.0 * f1x + 16.0 * f3x - 15.0 * f4x + 4.0 * f5x) * C2
        err_v = h * (-5.0 * f1v + 16.0 * f3v - 15.0 * f4v + 4.0 * f5v) * C2

        sc_x = atol + rtol * max(abs(x), abs(y5x))
        sc_v = atol + rtol * max(abs(v), abs(y5v))

        ex = abs(err_x) / sc_x
        ev = abs(err_v) / sc_v
        err_norm = ex if ex > ev else ev

        if err_norm < 1.0:
            t += h
            x = y5x
            v = y5v

            if err_norm == 0.0:
                factor = 5.0
            else:
                factor = 0.9 * err_norm ** (-0.2)
                if factor > 5.0:
                    factor = 5.0
            h_abs *= factor
        else:
            factor = 0.9 * err_norm ** (-0.2)
            if factor < 0.2:
                factor = 0.2
            h_abs *= factor

    return x, v


_vdp_integrate(1.0, 0.5, 0.0, 0.0, 1e-3)


def solve(problem):
    y0 = problem['y0']
    x0 = float(y0[0])
    v0 = float(y0[1])
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    mu = float(problem['mu'])

    x, v = _vdp_integrate(mu, x0, v0, t0, t1)
    return [x, v]
