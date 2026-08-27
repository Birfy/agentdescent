import numpy as np

def solve(problem: list[float]) -> list[float]:
    # Use numpy's polynomial root finder (companion matrix eigenvalues)
    roots = np.roots(problem)
    
    # All roots are real; discard any tiny imaginary parts and sort descending
    roots = np.sort(np.real(roots))[::-1]
    
    return roots.tolist()
