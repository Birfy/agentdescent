import numpy as np
from scipy.linalg import eig as scipy_eig

def solve(problem):
    # Convert to numpy array without copying if possible
    A = np.asarray(problem, dtype=float)
    
    # Use scipy's eig with finite check disabled for speed
    eigenvalues, eigenvectors = scipy_eig(A, check_finite=False)
    
    # Sort descending by real, then descending by imaginary
    order = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
    eigenvalues_sorted = eigenvalues[order]
    eigenvectors_sorted = eigenvectors[:, order]
    
    # Normalize columns
    norms = np.linalg.norm(eigenvectors_sorted, axis=0)
    norms[norms < 1e-12] = 1.0
    eigenvectors_sorted /= norms
    
    # Return transposed as list of lists
    return eigenvectors_sorted.T.tolist()
