import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def solve(problem: dict) -> dict:
    A = np.asarray(problem['A'], dtype=float)
    B = np.asarray(problem['B'], dtype=float)
    C = np.asarray(problem['C'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    x0 = np.asarray(problem['x_initial'], dtype=float)
    tau = float(problem['tau'])

    N, m = y.shape
    n = A.shape[0]
    p = B.shape[1]

    sqrt_tau = np.sqrt(tau)

    # Precompute pseudo-inverse and related products
    B_pinv = np.linalg.pinv(B)
    AB_pinv = B_pinv @ A
    C_sqrt = sqrt_tau * C

    # Build the sparse matrix M directly in CSR format using block structure
    # Total rows = N*p + N*m, columns = N*n
    # We'll use scipy.sparse.block_diag and manual placement for speed

    # Process noise block: N*p rows, N*n cols
    # Row block t: [0 ... -AB_pinv, B_pinv, 0 ...] for t=0..N-1
    # For t=0, the left part is empty, so just [B_pinv, 0...]
    # For t>0, left has (t-1)*n zeros, then -AB_pinv, then B_pinv, then (N-t-1)*n zeros

    # We can construct this efficiently using sparse.kron and diagonal shifts
    I_N = sparse.eye(N, format='csr')
    I_Nm1 = sparse.eye(N - 1, format='csr')

    # Main diagonal blocks: B_pinv for each row block
    main_diag = sparse.kron(I_N, sparse.csr_matrix(B_pinv), format='csr')

    # Subdiagonal blocks: -AB_pinv, placed one block row lower
    sub_diag = sparse.kron(I_Nm1, sparse.csr_matrix(-AB_pinv), format='csr')

    # Assemble M_dyn by stacking sub_diag into the correct position
    # Use vstack of block rows to keep CSR
    rows_dyn = []
    # First row: [B_pinv, 0, ...]
    rows_dyn.append(sparse.hstack([sparse.csr_matrix(B_pinv),
                                   sparse.csr_matrix((p, (N-1)*n))],
                                  format='csr'))
    # Middle rows: [0 ... -AB_pinv, B_pinv, 0 ...]
    for t in range(1, N):
        left = sparse.csr_matrix((p, (t-1)*n))
        right = sparse.csr_matrix((p, (N-t-1)*n))
        rows_dyn.append(sparse.hstack([left,
                                       sparse.csr_matrix(-AB_pinv),
                                       sparse.csr_matrix(B_pinv),
                                       right],
                                      format='csr'))
    M_dyn = sparse.vstack(rows_dyn, format='csr')

    # Measurement block: for t=1..N-1: -C_sqrt @ x_t = sqrt_tau * y_t
    # For t=0: x0 known, no column
    C_sp = sparse.csr_matrix(C_sqrt)
    M_meas = sparse.kron(I_Nm1, -C_sp, format='csr')
    # Place at rows m:end, cols 0:(N-1)*n
    M_meas_full = sparse.vstack([
        sparse.csr_matrix((m, N*n)),
        sparse.hstack([M_meas, sparse.csr_matrix(((N-1)*m, n))], format='csr')
    ], format='csr')

    M = sparse.vstack([M_dyn, M_meas_full], format='csr')

    # Build RHS vector b
    b = np.zeros(N*p + N*m)
    b[0:p] = -AB_pinv @ x0
    b[N*p:N*p+m] = sqrt_tau * (y[0] - C @ x0)
    for t in range(1, N):
        b[N*p + t*m : N*p + (t+1)*m] = sqrt_tau * y[t]

    # Solve normal equations
    MtM = (M.T @ M).tocsr()
    Mtb = M.T @ b
    x_vec = spsolve(MtM, -Mtb)

    x_hat = np.zeros((N + 1, n))
    x_hat[0] = x0
    x_hat[1:] = x_vec.reshape(N, n)

    # Recover w and v
    w_hat = np.zeros((N, p))
    v_hat = np.zeros((N, m))
    for t in range(N):
        w_hat[t] = B_pinv @ (x_hat[t+1] - A @ x_hat[t])
        v_hat[t] = y[t] - C @ x_hat[t]

    return {
        'x_hat': x_hat.tolist(),
        'w_hat': w_hat.tolist(),
        'v_hat': v_hat.tolist()
    }
