import numpy as np
import scipy.sparse

def solve(problem):
    try:
        data = np.asarray(problem['data'], dtype=np.float64)
        indices = np.asarray(problem['indices'], dtype=np.int32)
        indptr = np.asarray(problem['indptr'], dtype=np.int32)
        shape = tuple(problem['shape'])
        normed = problem['normed']
        n = shape[0]
    except Exception:
        return {'laplacian': {'data': [], 'indices': [], 'indptr': [], 'shape': problem.get('shape', (0, 0))}}

    try:
        A = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)
        A.sum_duplicates()

        if normed:
            degrees = np.zeros(n, dtype=np.float64)
            np.add.at(degrees, A.indices, A.data)
            
            mask = degrees > 0
            d_inv_sqrt = np.zeros(n, dtype=np.float64)
            d_inv_sqrt[mask] = np.sqrt(1.0 / degrees[mask])
            
            L = scipy.sparse.diags(degrees, format='csr') - A
            D_inv_sqrt = scipy.sparse.diags(d_inv_sqrt, format='csr')
            L = D_inv_sqrt @ L @ D_inv_sqrt
            
            if not scipy.sparse.isspmatrix_csr(L):
                L = L.tocsr()
            L.eliminate_zeros()
        else:
            degrees = np.zeros(n, dtype=np.float64)
            np.add.at(degrees, A.indices, A.data)
            
            L = A.copy()
            L.data *= -1.0
            L.setdiag(degrees)
            L.eliminate_zeros()
            
    except Exception:
        return {'laplacian': {'data': [], 'indices': [], 'indptr': [], 'shape': shape}}

    return {'laplacian': {
        'data': L.data.tolist(),
        'indices': L.indices.tolist(),
        'indptr': L.indptr.tolist(),
        'shape': L.shape
    }}
