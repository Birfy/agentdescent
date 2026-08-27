import numpy as np

def solve(problem: list[float]) -> list[float]:
    # Convert to numpy array and make monic
    coeffs = np.asarray(problem, dtype=np.float64)
    if coeffs[0] != 1.0:
        coeffs = coeffs / coeffs[0]
    
    # Use numpy.roots which is optimized for this
    roots = np.roots(coeffs)
    
    # All roots are real, take real part
    roots = np.real(roots)
    
    # Sort descending
    roots = -np.sort(-roots)
    
    return roots.tolist()
