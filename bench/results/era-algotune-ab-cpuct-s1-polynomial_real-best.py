import numpy as np

def solve(problem):
    coeffs = np.asarray(problem, dtype=np.float64)
    n = len(coeffs) - 1
    if n == 0:
        return []
    if n == 1:
        return [-coeffs[1] / coeffs[0]]
    
    # For high-degree polynomials with all real roots, use a more efficient method
    # than np.roots which computes all complex roots via eigendecomposition.
    # Since all roots are real, we can use the companion matrix with a real
    # eigensolver that exploits symmetry, or use Laguerre's method with deflation.
    # However, for n=396, the most reliable fast approach is to use np.roots
    # but with a slightly optimized path. Since np.roots is already quite optimized,
    # we'll keep it but avoid unnecessary overhead.
    
    # Use np.roots which is the standard high-precision method
    roots = np.roots(coeffs)
    # All roots are real, so take real part (should be exactly real but numerical errors)
    roots = np.real(roots)
    # Sort descending
    roots = np.sort(roots)[::-1]
    return roots.tolist()
