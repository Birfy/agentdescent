import numpy as np

def solve(problem):
    A = np.asarray(problem, dtype=float)
    n = A.shape[0]
    
    # Use LAPACK's dgeev via numpy - this is the core computation
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # For real matrices, eigenvalues come in conjugate pairs.
    # Sort by real desc, then imag desc using lexsort
    real_part = eigenvalues.real
    imag_part = eigenvalues.imag
    
    # np.lexsort sorts by last key first, so we pass (-imag, -real)
    order = np.lexsort((-imag_part, -real_part))
    
    sorted_eigenvalues = eigenvalues[order]
    sorted_eigenvectors = eigenvectors[:, order]
    
    # Eigenvectors from LAPACK are already normalized to unit norm,
    # but re-normalize to handle any numerical drift.
    # Compute norms efficiently using real/imag parts
    norms = np.sqrt(
        np.einsum('ij,ij->j', sorted_eigenvectors.real, sorted_eigenvectors.real) +
        np.einsum('ij,ij->j', sorted_eigenvectors.imag, sorted_eigenvectors.imag)
    )
    norms = np.maximum(norms, 1e-12)
    sorted_eigenvectors = sorted_eigenvectors / norms
    
    # Return as list of lists of complex numbers
    return sorted_eigenvectors.T.tolist()

PROMISE: 8
