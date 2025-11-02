import jax.numpy as jnp
import jax
from jax.scipy.optimize import curve_fit

@jax.jit
def biexponential_decay(t, A1, A2, tau1, tau2, c):
    return A1 * jnp.exp(-t/tau1) + A2 * jnp.exp(-t/tau2) + c

