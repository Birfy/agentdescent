import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    y0 = np.asarray(problem['y0'], dtype=np.float64)
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    F = float(problem['F'])
    n = y0.size

    def lorenz96(t, x):
        # Cyclic boundary conditions
        xm2 = np.empty(n, dtype=np.float64)
        xm1 = np.empty(n, dtype=np.float64)
        xp1 = np.empty(n, dtype=np.float64)

        xp1[:-1] = x[1:]
        xp1[-1] = x[0]

        xm1[1:] = x[:-1]
        xm1[0] = x[-1]

        xm2[2:] = x[:-2]
        xm2[0] = x[-2]
        xm2[1] = x[-1]

        return (xp1 - xm2) * xm1 - x + F

    sol = solve_ivp(
        lorenz96,
        (t0, t1),
        y0,
        method='RK45',
        rtol=1e-08,
        atol=1e-08
    )

    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    return sol.y[:, -1].tolist()
