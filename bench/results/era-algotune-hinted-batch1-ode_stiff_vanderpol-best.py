import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    """
    Solve the stiff Van der Pol equation using scipy's Radau IVP solver.
    Optimized by using a plain Python function for the RHS to minimize overhead
    inside the integrator's internal loops, avoiding NumPy array allocations.
    """
    y0 = problem['y0']
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    mu = float(problem['mu'])

    def vdp(t, y):
        x = y[0]
        v = y[1]
        return [v, mu * ((1.0 - x * x) * v - x)]

    sol = solve_ivp(
        vdp, [t0, t1], y0,
        method='Radau',
        rtol=1e-8,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(f'Solver failed: {sol.message}')

    return sol.y[:, -1].tolist()
