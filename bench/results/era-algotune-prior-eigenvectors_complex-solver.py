import numpy as np

def solve(problem):
    A = np.asarray(problem, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort indices by real part descending, then imaginary part descending
    order = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
    
    sorted_eigenvalues = eigenvalues[order]
    sorted_eigenvectors = eigenvectors[:, order].T
    
    # Normalize each eigenvector to unit norm in a vectorized way
    norms = np.linalg.norm(sorted_eigenvectors, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0  # avoid division by zero
    sorted_eigenvectors = sorted_eigenvectors / norms
    
    return sorted_eigenvectors.tolist()

PROMISE: 1
