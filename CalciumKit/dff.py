import jax
import jax.numpy as jnp

@jax.jit
def _mean_dff(signal):
    F0 = jnp.mean(signal, axis=-1, keepdims=False)
    F0 = jnp.where(F0 == 0, jnp.finfo(signal.dtype).eps, F0)
    return (signal - F0[..., None]) / F0[..., None] if signal.ndim == 2 else (signal - F0) / F0

@jax.jit
def _median_dff(signal):
    F0 = jnp.median(signal, axis=-1, keepdims=False)
    F0 = jnp.where(F0 == 0, jnp.finfo(signal.dtype).eps, F0)
    return (signal - F0[..., None]) / F0[..., None] if signal.ndim == 2 else (signal - F0) / F0