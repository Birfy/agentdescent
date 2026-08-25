import numpy as np

def solve(problem):
    A = np.ascontiguousarray(problem['A'], dtype=np.float64)
    evals, evecs = np.linalg.eigh(A)
    np.maximum(evals, 0.0, out=evals)
    np.sqrt(evals, out=evals)
    evecs *= evals
    X = evecs @ evecs.T
    return {'X': X}
