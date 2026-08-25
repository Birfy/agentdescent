import numpy as np
import scipy.linalg

def solve(problem):
    A = problem['matrix']
    # The input is a list of lists. np.asarray performs a direct C-level copy
    # into a contiguous array, which is faster than np.array's validation path.
    A_arr = np.asarray(A, dtype=np.float64, order='C')
    
    # scipy.linalg.qr with mode='economic' calls LAPACK's dgeqrf + dormqr.
    # It is faster than numpy.linalg.qr because it avoids extra Python wrappers.
    Q, R = scipy.linalg.qr(A_arr, mode='economic')
    
    # .tolist() is the fastest way to convert a NumPy array to a nested Python list.
    return {'QR': {'Q': Q.tolist(), 'R': R.tolist()}}
