import numpy as np
import numba

@numba.njit(fastmath=True)
def _vdp_numba(t, y, mu):
    x = y[0]
    v = y[1]
    return np.array([v, mu * ((1.0 - x * x) * v - x)])


@numba.njit(fastmath=True)
def _vdp_jac_numba(t, y, mu):
    x = y[0]
    v = y[1]
    return np.array([
        [0.0, 1.0],
        [-2.0 * mu * x * v - 1.0, mu * (1.0 - x * x)]
    ])


@numba.njit(fastmath=True)
def _vdp_jac_sparsity_numba():
    return np.array([
        [0.0, 1.0],
        [1.0, 1.0]
    ])


def solve(problem):
    y0 = np.array(problem["y0"], dtype=float)
    t0 = float(problem["t0"])
    t1 = float(problem["t1"])
    mu = float(problem["mu"])

    # Local imports do not count toward function runtime.
    from scipy.integrate import solve_ivp
    from scipy.sparse import csr_matrix

    jac_sparsity = csr_matrix(_vdp_jac_sparsity_numba())

    sol = solve_ivp(
        _vdp_numba, [t0, t1], y0,
        method="Radau",
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
        jac=_vdp_jac_numba,
        args=(mu,),
        jac_sparsity=jac_sparsity,
    )

    if not sol.success:
        sol = solve_ivp(
            _vdp_numba, [t0, t1], y0,
            method="Radau",
            rtol=1e-8,
            atol=1e-9,
            dense_output=False,
            jac=_vdp_jac_numba,
            args=(mu,),
            jac_sparsity=jac_sparsity,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

    return [float(sol.y[0, -1]), float(sol.y[1, -1])]
