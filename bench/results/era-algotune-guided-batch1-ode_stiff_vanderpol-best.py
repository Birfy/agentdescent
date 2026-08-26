import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    mu = float(problem['mu'])
    y0 = problem['y0']
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    
    def fun(t, y):
        return np.array([y[1], mu * ((1.0 - y[0] * y[0]) * y[1] - y[0])])
    
    def jac(t, y):
        return np.array([[0.0, 1.0], [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] * y[0])]])
    
    sol = solve_ivp(
        fun,
        (t0, t1),
        y0,
        method='Radau',
        rtol=1e-08,
        atol=1e-09,
        dense_output=False,
        jac=jac
    )
    
    return sol.y[:, -1].tolist()
