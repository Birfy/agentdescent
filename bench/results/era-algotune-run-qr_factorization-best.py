import numpy as np

def solve(problem):
    A = np.ascontiguousarray(problem['matrix'], dtype=np.float64)
    Q, R = np.linalg.qr(A, mode='reduced')
    return {'QR': {'Q': Q.tolist(), 'R': R.tolist()}}
