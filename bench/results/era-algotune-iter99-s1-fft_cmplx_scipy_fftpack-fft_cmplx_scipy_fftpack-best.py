import jax
import jax.numpy as jnp
import numpy as np

# JIT compile the FFT function
@jax.jit
def _fftn(x):
    return jnp.fft.fftn(x)

def solve(problem):
    # Convert to JAX array (if not already) and compute FFT
    arr = jnp.asarray(problem)
    result = _fftn(arr)
    # Convert back to numpy (or keep as numpy array for the harness)
    return np.asarray(result)
