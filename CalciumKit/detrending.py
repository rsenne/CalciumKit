import jax.numpy as jnp
import jax
from jax.scipy.optimize import curve_fit

@jax.jit
def biexponential_decay(t, A1, A2, tau1, tau2, c):
    return A1 * jnp.exp(-t/tau1) + A2 * jnp.exp(-t/tau2) + c

@jax.jit
def huber_loss(residuals, delta):
    abs_r = jnp.abs(residuals)
    quadratic = 0.5 * residuals**2
    linear = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quadratic, linear)