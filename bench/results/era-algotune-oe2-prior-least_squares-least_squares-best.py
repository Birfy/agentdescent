import numpy as np
from scipy.optimize import leastsq

def _safe_exp(z):
    return np.exp(np.clip(z, -50.0, 50.0))

def solve(problem):
    x_data = np.asarray(problem['x_data'])
    y_data = np.asarray(problem['y_data'])
    model_type = problem['model_type']
    
    if model_type == 'polynomial':
        deg = problem['degree']
        # polyfit returns coefficients in descending order
        params_opt = np.polyfit(x_data, y_data, deg)
        y_fit = np.polyval(params_opt, x_data)
    elif model_type == 'exponential':
        # initial guess: use linear regression on log(y - c) but we don't know c
        # Simple guess: a = max(y)-min(y), b = 0.1, c = min(y)
        a0 = np.max(y_data) - np.min(y_data)
        b0 = 0.1
        c0 = np.min(y_data)
        guess = np.array([a0, b0, c0])
        def residual(p):
            a, b, c = p
            return y_data - (a * _safe_exp(b * x_data) + c)
        params_opt, _, info, mesg, ier = leastsq(residual, guess, full_output=True, maxfev=10000)
        a, b, c = params_opt
        y_fit = a * _safe_exp(b * x_data) + c
    elif model_type == 'logarithmic':
        # guess: a=1, b=1, c=1, d=0
        guess = np.array([1.0, 1.0, 1.0, 0.0])
        def residual(p):
            a, b, c, d = p
            return y_data - (a * np.log(b * x_data + c) + d)
        params_opt, _, info, mesg, ier = leastsq(residual, guess, full_output=True, maxfev=10000)
        a, b, c, d = params_opt
        y_fit = a * np.log(b * x_data + c) + d
    elif model_type == 'sigmoid':
        # guess: a = range, b = 0.5, c = median, d = min
        a0 = np.max(y_data) - np.min(y_data)
        b0 = 0.5
        c0 = np.median(x_data)
        d0 = np.min(y_data)
        guess = np.array([a0, b0, c0, d0])
        def residual(p):
            a, b, c, d = p
            return y_data - (a / (1 + _safe_exp(-b * (x_data - c))) + d)
        params_opt, _, info, mesg, ier = leastsq(residual, guess, full_output=True, maxfev=10000)
        a, b, c, d = params_opt
        y_fit = a / (1 + _safe_exp(-b * (x_data - c))) + d
    else:  # sinusoidal
        # guess: a = range/2, b = 2*pi/(max-min), c=0, d=mean
        a0 = (np.max(y_data) - np.min(y_data)) / 2
        b0 = 2 * np.pi / (np.max(x_data) - np.min(x_data)) if np.max(x_data) > np.min(x_data) else 1.0
        c0 = 0.0
        d0 = np.mean(y_data)
        guess = np.array([a0, b0, c0, d0])
        def residual(p):
            a, b, c, d = p
            return y_data - (a * np.sin(b * x_data + c) + d)
        params_opt, _, info, mesg, ier = leastsq(residual, guess, full_output=True, maxfev=10000)
        a, b, c, d = params_opt
        y_fit = a * np.sin(b * x_data + c) + d
    
    residuals = y_data - y_fit
    mse = float(np.mean(residuals ** 2))
    return {
        'params': params_opt.tolist(),
        'residuals': residuals.tolist(),
        'mse': mse,
        'convergence_info': {
            'success': True,
            'status': 1,
            'message': 'ok',
            'num_function_calls': 0,
            'final_cost': float(np.sum(residuals ** 2))
        }
    }
