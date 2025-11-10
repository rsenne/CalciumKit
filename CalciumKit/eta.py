import jax
import jax.numpy as jnp

@jax.jit
def _simple_eta_core(signal, t, lags, events):
    """
    signal: (T,)
    t:      (T,)
    lags:   (W,)   relative times (window grid)
    events: (E,)   event times (same units as t)
    """
    # absolute sample times per event
    sample_times = events[:, None] + lags[None, :]          # (E, W)

    # Bounds
    tmin = t[0]
    tmax = t[-1]

    # Vectorized 1D linear interpolation with *clipping* + mask
    def interp_at(times_1d):
        within = (times_1d >= tmin) & (times_1d <= tmax)
        tq = jnp.clip(times_1d, tmin, tmax)
        vals = jnp.interp(tq, t, signal)                   # no left/right NaNs here
        return jnp.where(within, vals, jnp.nan)

    traces = jax.vmap(interp_at)(sample_times)              # (E, W)
    avg = jnp.nanmean(traces, axis=0)                       # (W,)
    return avg, traces

def simple_eta(signal, t, window, events, *, dt=None, num=None, return_traces=False):
    """
    Compute a simple event-triggered average by aligning `signal` to `events`
    within `window` relative to each event, using linear interpolation.

    Args
    ----
    signal : (T,) array
        Signal of interest, sampled at times `t`.
    t : (T,) array
        Monotonic time stamps corresponding to `signal`.
    window : (t_start, t_end)
        Relative time window around each event (e.g., (-0.5, 1.0)).
    events : (E,) array
        Event times (same units as `t`).
    dt : float, optional
        Desired sampling step for the window grid. If not provided,
        it defaults to `t[1] - t[0]`.
    num : int, optional
        Number of samples in the window grid. If provided, overrides `dt`.
    return_traces : bool, default False
        If True, also return the (E, W) stack of aligned traces.

    Returns
    -------
    avg : (W,) array
        Event-triggered average over the window grid.
    lags : (W,) array
        Relative time axis for the window.
    traces : (E, W) array, optional
        Per-event aligned traces (with NaNs where the window exceeds data).
    """
    t0, t1 = float(window[0]), float(window[1])
    if num is None:
        if dt is None:
            # assume near-uniform sampling; use first step
            dt = float(t[1] - t[0])
        # ensure inclusive of endpoints
        num = int(jnp.floor((t1 - t0) / dt)) + 1
        num = max(num, 1)
    lags = jnp.linspace(t0, t1, num)
    avg, traces = _simple_eta_core(jnp.asarray(signal), jnp.asarray(t), lags, jnp.asarray(events))
    return (avg, lags, traces) if return_traces else (avg, lags)
