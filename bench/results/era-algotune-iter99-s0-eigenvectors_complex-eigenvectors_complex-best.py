import numpy as np
from numpy.typing import NDArray

def solve(problem: NDArray) -> list[list[complex]]:
    A = np.asarray(problem)
    # Use the fastest available eigen solver for general real matrices
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort by descending real part, then descending imaginary part
    order = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Normalize eigenvectors to unit Euclidean norm
    norms = np.linalg.norm(eigenvectors, axis=0)
    # Avoid division by zero for degenerate cases
    norms[norms < 1e-12] = 1.0
    eigenvectors = eigenvectors / norms
    
    return eigenvectors.T.tolist()
