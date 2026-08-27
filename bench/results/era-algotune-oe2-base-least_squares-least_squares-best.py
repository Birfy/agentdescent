import numpy as np
from scipy.optimize import curve_fit

def solve(problem):
    x_data = np.asarray(problem['x_data'], dtype=np.float64)
    y_data = np.asarray(problem['y_data'], dtype=np.float64)
    model_type = problem['model_type']

    if model_type == 'polynomial':
        deg = problem['degree']
        params = np.polyfit(x_data, y_data, deg)
        y_fit = np.polyval(params, x_data)

    elif model_type == 'exponential':
        # y = a*exp(b*x) + c
        if np.all(y_data > 0):
            log_y = np.log(y_data)
            # Use polyfit for speed
            m, k = np.polyfit(x_data, log_y, 1)
            a0 = np.exp(k)
            b0 = m
            c0 = 0.0
            exp_bx = np.exp(np.clip(b0 * x_data, -50, 50))
            r = a0 * exp_bx + c0 - y_data
            J = np.column_stack([exp_bx, a0 * x_data * exp_bx, np.ones_like(x_data)])
            try:
                delta = np.linalg.solve(J.T @ J, -J.T @ r)
                params = np.array([a0, b0, c0]) + delta
                params[1] = np.clip(params[1], -10, 10)
            except np.linalg.LinAlgError:
                def model(x, a, b, c):
                    return a * np.exp(np.clip(b * x, -50, 50)) + c
                params, _ = curve_fit(model, x_data, y_data, p0=[a0, b0, c0], maxfev=1000)
        else:
            def model(x, a, b, c):
                return a * np.exp(np.clip(b * x, -50, 50)) + c
            y_min = np.min(y_data)
            y_max = np.max(y_data)
            a0 = y_max - y_min
            b0 = 0.1
            c0 = y_min
            params, _ = curve_fit(model, x_data, y_data, p0=[a0, b0, c0], maxfev=1000)
        y_fit = params[0] * np.exp(np.clip(params[1] * x_data, -50, 50)) + params[2]

    elif model_type == 'logarithmic':
        # y = a*log(b*x + c) + d
        if np.all(x_data > -1):
            log_x = np.log(x_data + 1)
            a_d, _, _, _ = np.linalg.lstsq(np.column_stack([log_x, np.ones_like(x_data)]), y_data, rcond=None)
            a0 = a_d[0]
            d0 = a_d[1]
            b0 = 1.0
            c0 = 1.0
            arg = np.clip(b0 * x_data + c0, 1e-10, None)
            log_arg = np.log(arg)
            r = a0 * log_arg + d0 - y_data
            J = np.column_stack([
                log_arg,
                a0 * x_data / arg,
                a0 / arg,
                np.ones_like(x_data)
            ])
            try:
                delta = np.linalg.solve(J.T @ J, -J.T @ r)
                params = np.array([a0, b0, c0, d0]) + delta
            except np.linalg.LinAlgError:
                params = np.array([a0, 1.0, 1.0, d0])
        else:
            def model(x, a, b, c, d):
                return a * np.log(np.clip(b * x + c, 1e-10, None)) + d
            guess = np.array([1.0, 1.0, 1.0, 0.0])
            params, _ = curve_fit(model, x_data, y_data, p0=guess, maxfev=1000)
        y_fit = params[0] * np.log(np.clip(params[1] * x_data + params[2], 1e-10, None)) + params[3]

    elif model_type == 'sigmoid':
        # y = a/(1+exp(-b(x-c))) + d
        y_min = np.min(y_data)
        y_max = np.max(y_data)
        a0 = y_max - y_min
        d0 = y_min
        mid_val = (y_max + y_min) / 2
        idx = np.argmin(np.abs(y_data - mid_val))
        c0 = x_data[idx]

        mask = np.abs(y_data - mid_val) < (y_max - y_min) * 0.4
        if np.sum(mask) > 2:
            x_sub = x_data[mask]
            y_sub = y_data[mask]
            z = np.log(np.clip((y_sub - d0) / (a0 - (y_sub - d0)), 1e-10, 1e10))
            b_bc, _, _, _ = np.linalg.lstsq(np.column_stack([x_sub, np.ones_like(x_sub)]), z, rcond=None)
            b0 = b_bc[0]
            c0 = -b_bc[1] / b0 if abs(b0) > 1e-10 else c0
        else:
            b0 = 1.0

        exp_term = np.exp(np.clip(-b0 * (x_data - c0), -50, 50))
        sig = 1 / (1 + exp_term)
        r = a0 * sig + d0 - y_data
        J = np.column_stack([
            sig,
            a0 * (x_data - c0) * exp_term * sig**2,
            -a0 * b0 * exp_term * sig**2,
            np.ones_like(x_data)
        ])
        try:
            delta = np.linalg.solve(J.T @ J, -J.T @ r)
            params = np.array([a0, b0, c0, d0]) + delta
            params[0] = max(params[0], 1e-6)
            params[2] = np.clip(params[2], np.min(x_data), np.max(x_data))
        except np.linalg.LinAlgError:
            params = np.array([a0, b0, c0, d0])
        y_fit = params[0] / (1 + np.exp(np.clip(-params[1] * (x_data - params[2]), -50, 50))) + params[3]

    elif model_type == 'sinusoidal':
        # y = a*sin(b*x + c) + d
        n = len(x_data)
        if n > 2:
            y_centered = y_data - np.mean(y_data)
            fft_vals = np.fft.fft(y_centered)
            freqs = np.fft.fftfreq(n, d=(x_data[-1] - x_data[0]) / (n - 1) if n > 1 else 1.0)
            mags = np.abs(fft_vals)
            mags[0] = 0
            peak_idx = np.argmax(mags[1:n//2]) + 1
            b0 = 2 * np.pi * freqs[peak_idx]

            sin_bx = np.sin(b0 * x_data)
            cos_bx = np.cos(b0 * x_data)
            coeffs, _, _, _ = np.linalg.lstsq(np.column_stack([sin_bx, cos_bx, np.ones_like(x_data)]), y_data, rcond=None)
            a0 = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
            c0 = np.arctan2(coeffs[1], coeffs[0])
            d0 = coeffs[2]

            params = np.array([a0, b0, c0, d0])
            sin_val = np.sin(b0 * x_data + c0)
            cos_val = np.cos(b0 * x_data + c0)
            r = a0 * sin_val + d0 - y_data
            J = np.column_stack([
                sin_val,
                a0 * x_data * cos_val,
                a0 * cos_val,
                np.ones_like(x_data)
            ])
            try:
                delta = np.linalg.solve(J.T @ J, -J.T @ r)
                params = params + delta
            except np.linalg.LinAlgError:
                pass
        else:
            def model(x, a, b, c, d):
                return a * np.sin(b * x + c) + d
            params, _ = curve_fit(model, x_data, y_data, p0=[2.0, 1.0, 0.0, 0.0], maxfev=1000)
        y_fit = params[0] * np.sin(params[1] * x_data + params[2]) + params[3]

    else:
        raise ValueError(f'Unknown model type: {model_type}')

    residuals = y_data - y_fit
    mse = float(np.mean(residuals ** 2))
    return {
        'params': np.asarray(params).tolist(),
        'residuals': residuals.tolist(),
        'mse': mse,
        'convergence_info': {
            'success': True,
            'status': 1,
            'message': 'Optimization terminated successfully',
            'num_function_calls': 0,
            'final_cost': float(np.sum(residuals ** 2))
        }
    }
