import numpy as np

def solve(problem):
    A = np.asarray(problem['A'], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(A)
    np.maximum(eigvals, 0.0, out=eigvals)
    np.sqrt(eigvals, out=eigvals)
    eigvecs *= eigvals
    X = eigvecs @ eigvecs.T
    return {'X': X}
