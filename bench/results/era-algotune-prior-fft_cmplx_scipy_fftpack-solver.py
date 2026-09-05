import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

# Pre-compile for a typical size? But size is fixed at 1860. We can compile on first call.
# But we can pre-compile with a dummy array of shape (1860,1860) at module load.
# However, the problem might have different sizes? The task says n=1860, so fixed.

# We'll define a jitted function that takes a 2D complex array.
@jit
def _fftn_jax(x):
    return jnp.fft.fftn(x)

# To avoid recompilation for different shapes, we can use jax.jit with static_argnums? Not needed.

def solve(problem):
    # Convert to jax array (zero-copy if possible)
    x = jnp.asarray(problem)
    # Call jitted function
    y = _fftn_jax(x)
    # Convert back to numpy (copy)
    return np.asarray(y)
