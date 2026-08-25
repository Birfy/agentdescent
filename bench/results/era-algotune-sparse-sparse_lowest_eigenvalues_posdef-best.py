from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def solve(problem: dict[str, Any]) -> list[float]:
    mat: sparse.spmatrix = problem['matrix'].asformat('csr')
    k: int = int(problem['k'])
    n = mat.shape[0]
    if k >= n or n < 2 * k + 1:
        vals = np.linalg.eigvalsh(mat.toarray())
        return [float(v) for v in vals[:k]]
    try:
        vals = eigsh(mat, k=k, which='SM', return_eigenvectors=False, maxiter=n * 200, ncv=min(n - 1, max(2 * k + 1, 20)))
    except Exception:
        vals = np.linalg.eigvalsh(mat.toarray())[:k]
    return [float(v) for v in np.sort(np.real(vals))]
