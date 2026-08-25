import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    y0 = np.asarray(problem['y0'], dtype=np.float64)
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    F = float(problem['F'])
    N = y0.shape[0]

    if N == 0:
        return []

    def lorenz96(t, x):
        x_ip1 = np.empty(N, dtype=np.float64)
        x_im1 = np.empty(N, dtype=np.float64)
        x_im2 = np.empty(N, dtype=np.float64)

        x_ip1[:-1] = x[1:]
        x_ip1[-1] = x[0]

        x_im1[1:] = x[:-1]
        x_im1[0] = x[-1]

        x_im2[2:] = x[:-2]
        x_im2[0] = x[-2]
        x_im2[1] = x[-1]

        return (x_ip1 - x_im2) * x_im1 - x + F

    sol = solve_ivp(
        lorenz96, [t0, t1], y0,
        method='RK45', rtol=1e-08, atol=1e-08,
        dense_output=False, t_eval=None
    )

    if sol.success:
        return sol.y[:, -1].tolist()
    else:
        raise RuntimeError(f'Solver failed: {sol.message}')
