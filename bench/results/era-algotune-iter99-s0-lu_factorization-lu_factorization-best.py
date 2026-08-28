import numpy as np
from scipy.linalg import lu as scipy_lu
from numba import njit, prange

def solve(problem):
    A = np.array(problem['matrix'], dtype=float)
    n = A.shape[0]
    
    # Use scipy's LU as a baseline for correctness
    P, L, U = scipy_lu(A, permute_l=False, check_finite=False)
    
    # For large n, scipy's lu is already well-optimized.
    # We can try to use a faster path for the common case:
    # If the matrix is already in a form where no pivoting is needed,
    # we could use a custom LU without pivoting. But detecting that
    # costs time, so we stick with scipy's implementation.
    
    # However, we can try to use the fact that the input is a list of lists
    # and convert to numpy more efficiently if needed. But np.array is fine.
    
    return {'LU': {'P': P, 'L': L, 'U': U}}

# Alternative approach: try using a custom compiled LU via numba
# This might be faster for very large matrices if we can avoid overhead.
# But scipy's lu uses LAPACK which is highly optimized, so likely best.

# Let's try a different approach: use scipy.linalg.lu_factor and lu_solve
# but we need the full P, L, U. lu_factor gives pivots, we can reconstruct.

# Actually, let's try to use a custom LU with partial pivoting in numba
# for potential speedup, but need to be careful about numerical stability.

# For now, keep the scipy version as it's already quite fast.
