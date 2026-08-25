import numpy as np
import scipy.sparse

def solve(problem):
    data = np.asarray(problem['data'], dtype=np.float64)
    indices = np.asarray(problem['indices'], dtype=np.int32)
    indptr = np.asarray(problem['indptr'], dtype=np.int32)
    shape = tuple(problem['shape'])
    n = shape[0]
    normed = problem['normed']

    if n == 0:
        return {'laplacian': {'data': [], 'indices': [], 'indptr': [0], 'shape': (0, 0)}}

    A = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    degrees = np.asarray(A.sum(axis=1)).ravel()

    if normed:
        with np.errstate(divide='ignore', invalid='ignore'):
            d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        L = scipy.sparse.diags(d_inv_sqrt, format='csr') @ A @ scipy.sparse.diags(d_inv_sqrt, format='csr')
        L = scipy.sparse.eye(n, format='csr') - L
    else:
        L = scipy.sparse.diags(degrees, format='csr') - A

    if not isinstance(L, scipy.sparse.csr_matrix):
        L = L.tocsr()

    L.eliminate_zeros()

    return {
        'laplacian': {
            'data': L.data.tolist(),
            'indices': L.indices.tolist(),
            'indptr': L.indptr.tolist(),
            'shape': L.shape
        }
    }
