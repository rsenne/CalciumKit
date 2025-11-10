import jax
import jax.numpy as jnp

@jax.jit
def _mean_dff(signal):
    """Simple mean ΔF/F baseline."""
    F0 = jnp.mean(signal, axis=-1, keepdims=False)
    F0 = jnp.where(F0 == 0, jnp.finfo(signal.dtype).eps, F0)
    return (signal - F0[..., None]) / F0[..., None] if signal.ndim == 2 else (signal - F0) / F0

@jax.jit
def _median_dff(signal):
    """Simple dF/f using median as baseline F0."""
    F0 = jnp.median(signal, axis=-1, keepdims=False)
    F0 = jnp.where(F0 == 0, jnp.finfo(signal.dtype).eps, F0)
    return (signal - F0[..., None]) / F0[..., None] if signal.ndim == 2 else (signal - F0) / F0

@jax.jit
def _rolling_percentile_dff(signal, percentile: float = 8.0, window: int = 201, pad_mode: str = "reflect"):
    """
    Rolling percentile ΔF/F baseline.
    Args:
        signal: array of shape (T,) or (N, T), time on the last axis.
        percentile: e.g. 8.0 for the 8th percentile.
        window: rolling window length (int). Can be even or odd.
        pad_mode: padding mode passed to jnp.pad, e.g. "reflect" or "edge".
    Returns:
        dFF array with same shape as `signal`.
    """
    # Compute asymmetric left/right pad so output length matches input for any window size
    left = window // 2
    right = window - 1 - left

    # Build pad widths (no pad on non-time axes)
    pad_width = [(0, 0)] * (signal.ndim - 1) + [(left, right)]
    x = jnp.pad(signal, pad_width, mode=pad_mode)

    # Sliding windows over the last axis -> shape (..., T, window)
    windows = jnp.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=-1)

    # Rolling baseline via percentile along the window axis
    F0 = jnp.percentile(windows, percentile, axis=-1)

    # Guard against divide-by-zero
    F0 = jnp.where(F0 == 0, jnp.finfo(signal.dtype).eps, F0)

    # ΔF/F with broadcasting over the last axis
    return (signal - F0) / F0

def dff(signal: jnp.ndarray, method: str = "rolling_percentile", **kwargs) -> jnp.ndarray:
    """
    Compute ΔF/F for fluorescence signal using specified baseline method.

    Args:
        signal: array of shape (T,) or (N, T), time on the last axis.
        method: baseline method, one of "mean", "median", "rolling_percentile".
        **kwargs: additional parameters for the chosen method.

    Returns:
        dFF array with same shape as `signal`.
    """
    if method == "mean":
        return _mean_dff(signal)
    elif method == "median":
        return _median_dff(signal)
    elif method == "rolling_percentile":
        return _rolling_percentile_dff(signal, **kwargs)
    else:
        raise ValueError(f"Unknown dFF method: {method}. Accepted methods are 'mean', 'median', 'rolling_percentile'.")
