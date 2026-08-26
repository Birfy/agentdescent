import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    """
    Solve the stiff Van der Pol equation.
    """
    y0 = np.asarray(problem['y0'], dtype=np.float64)
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    mu = float(problem['mu'])

    def vdp(t, y):
        x = y[0]
        v = y[1]
        return [v, mu * ((1.0 - x * x) * v - x)]

    sol = solve_ivp(
        vdp,
        [t0, t1],
        y0,
        method='Radau',
        rtol=1e-08,
        atol=1e-09,
        dense_output=False,
        t_eval=None
    )

    if sol.success:
        return sol.y[:, -1].tolist()
    else:
        raise RuntimeError(f'Solver failed: {sol.message}')
