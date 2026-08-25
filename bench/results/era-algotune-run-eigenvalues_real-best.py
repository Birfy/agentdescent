import numpy as np
from numpy.typing import NDArray

def solve(problem: NDArray) -> list[float]:
    """
    Solve the eigenvalues problem for the given symmetric matrix.
    The solution returned is a list of eigenvalues in descending order.

    :param problem: A symmetric numpy matrix.
    :return: List of eigenvalues in descending order.
    """
    A = np.ascontiguousarray(problem, dtype=np.float64)
    w = np.linalg.eigvalsh(A)
    w[::-1].sort()
    return w.tolist()
