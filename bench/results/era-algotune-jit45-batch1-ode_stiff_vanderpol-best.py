import numpy as np
from scipy.integrate import solve_ivp

def solve(problem):
    mu = float(problem['mu'])
    y0 = np.asarray(problem['y0'], dtype=float)
    t0 = float(problem['t0'])
    t1 = float(problem['t1'])
    
    def vdp(t, y):
        x = y[0]
        v = y[1]
        return np.array([v, mu * ((1.0 - x * x) * v - x)])
        
    sol = solve_ivp(
        vdp,
        (t0, t1),
        y0,
        method='Radau',
        rtol=1e-08,
        atol=1e-09,
        dense_output=False,
        jac_sparsity=np.array([[False, True], [True, True]])
    )
    
    if not sol.success:
        raise RuntimeError(f'Solver failed: {sol.message}')
        
    return sol.y[:, -1].tolist()
