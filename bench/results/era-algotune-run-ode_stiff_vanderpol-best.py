import numpy as np
from scipy.integrate._ivp.radau import Radau


def solve(problem):
    mu = float(problem['mu'])
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    y0 = np.asarray(problem['y0'], dtype=float)

    def fun(t, y):
        return np.array(
            [y[1], mu * ((1.0 - y[0] * y[0]) * y[1] - y[0])],
            dtype=float
        )

    def jac(t, y):
        x = y[0]
        v = y[1]
        return np.array(
            [[0.0, 1.0],
             [-2.0 * mu * x * v - 1.0, mu * (1.0 - x * x)]],
            dtype=float
        )

    radau = Radau(fun, t0, y0, t1, rtol=1e-08, atol=1e-09, jac=jac)
    radau.step()
    while radau.status == 'running':
        radau.step()

    return radau.y.tolist()
