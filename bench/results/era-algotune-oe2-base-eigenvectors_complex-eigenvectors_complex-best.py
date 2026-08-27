import numpy as np
from scipy.linalg import eig

def solve(problem):
    # Compute eigenvalues and eigenvectors using scipy
    eigenvalues, eigenvectors = eig(problem)
    
    # Sort by real part descending, then imaginary part descending
    # Use a lexicographic sort: first on -real, then on -imag
    order = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Normalize each eigenvector column
    norms = np.linalg.norm(eigenvectors, axis=0)
    # Avoid division by zero
    norms = np.where(norms < 1e-12, 1.0, norms)
    eigenvectors = eigenvectors / norms
    
    # Transpose to get list of eigenvectors as rows
    return eigenvectors.T.tolist()
