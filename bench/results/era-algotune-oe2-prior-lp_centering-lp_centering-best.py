import logging
from typing import Any
import cvxpy as cp
import numpy as np

def solve(problem: dict[str, Any]) -> dict[str, list]:
    """
        Solve the lp centering problem using CVXPY.

        :param problem: A dictionary of the lp centering problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp centering problem.
        """
    c = np.array(problem['c'])
    A = np.array(problem['A'])
    b = np.array(problem['b'])
    n = c.shape[0]
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c.T @ x - cp.sum(cp.log(x))), [A @ x == b])
    prob.solve(solver='CLARABEL')
    assert prob.status == 'optimal'
    return {'solution': x.value.tolist()}
