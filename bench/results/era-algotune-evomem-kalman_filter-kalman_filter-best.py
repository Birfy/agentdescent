import numpy as np

def solve(problem: dict) -> dict:
    A = np.asarray(problem["A"], dtype=float)
    B = np.asarray(problem["B"], dtype=float)
    C = np.asarray(problem["C"], dtype=float)
    y = np.asarray(problem["y"], dtype=float)
    x0 = np.asarray(problem["x_initial"], dtype=float)
    tau = float(problem["tau"])

    N, m = y.shape
    n = A.shape[0]
    p = B.shape[1]

    Q = B @ B.T
    R = np.eye(m) / tau

    C_T = C.T
    A_T = A.T
    I_n = np.eye(n)
    I_m = np.eye(m)

    # Forward pass: Kalman filter
    x_filt = np.zeros((N + 1, n))
    P_filt = np.zeros((N + 1, n, n))

    x_prior = x0.copy()
    P_prior = np.zeros((n, n))

    for t in range(N):
        # S = C P C^T + R
        S = C @ P_prior @ C_T + R
        # Solve for K efficiently
        K = np.linalg.solve(S.T, (P_prior @ C_T).T).T  # K = P C^T S^{-1}
        innovation = y[t] - C @ x_prior
        x_post = x_prior + K @ innovation
        P_post = P_prior - K @ S @ K.T
        P_post = (P_post + P_post.T) / 2

        x_filt[t] = x_post
        P_filt[t] = P_post

        x_prior = A @ x_post
        P_prior = A @ P_post @ A_T + Q

    x_filt[N] = x_prior
    P_filt[N] = P_prior

    # Backward RTS smoother
    x_smooth = np.zeros((N + 1, n))
    x_smooth[N] = x_filt[N]

    for t in range(N - 1, -1, -1):
        P_pred = A @ P_filt[t] @ A_T + Q
        # G = P_filt A^T P_pred^{-1}
        G = np.linalg.solve(P_pred.T, (P_filt[t] @ A_T).T).T
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - A @ x_filt[t])

    # Noise estimates
    B_pinv = np.linalg.pinv(B)
    diff = x_smooth[1:] - (A @ x_smooth[:-1].T).T
    w_hat = (B_pinv @ diff.T).T
    v_hat = y - (C @ x_smooth[:-1].T).T

    return {
        "x_hat": x_smooth.tolist(),
        "w_hat": w_hat.tolist(),
        "v_hat": v_hat.tolist(),
    }
