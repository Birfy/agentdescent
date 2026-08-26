import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    mu = float(problem['mu'])
    y0 = np.array(problem['y0'], dtype=np.float64)
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])

    def vdp(t, y):
        return np.array((y[1], mu * ((1.0 - y[0] * y[0]) * y[1] - y[0])))

    def jac(t, y):
        return np.array(
            (
                (0.0, 1.0),
                (-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] * y[0]))
            )
        )

    sol = solve_ivp(
        vdp,
        (t0, t1),
        y0,
        method='Radau',
        rtol=1e-06,
        atol=1e-08,
        dense_output=False,
        jac=jac
    )
    
    if sol.success:
        return sol.y[:, -1].tolist()
    else:
        raise RuntimeError(f"Solver failed: {sol.message}")
