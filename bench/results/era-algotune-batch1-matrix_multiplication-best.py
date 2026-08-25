import numpy as np

def solve(problem):
    A = problem['A']
    B = problem['B']
    
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    
    return np.dot(A_arr, B_arr)
