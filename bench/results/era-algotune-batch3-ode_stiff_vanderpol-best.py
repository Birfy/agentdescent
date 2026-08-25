import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    mu = float(problem['mu'])
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    y0 = np.asarray(problem['y0'], dtype=float)

    def vdp_jac(t, y):
        x, v = y
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * x * v - 1.0, mu * (1.0 - x * x)]
        ])

    sol = solve_ivp(
        lambda t, y: np.array([y[1], mu * ((1.0 - y[0] * y[0]) * y[1] - y[0])]),
        [t0, t1],
        y0,
        method='Radau',
        rtol=1e-08,
        atol=1e-09,
        jac=vdp_jac,
        dense_output=False
    )

    if not sol.success:
        raise RuntimeError(f'Solver failed: {sol.message}')
    
    return sol.y[:, -1].tolist()
