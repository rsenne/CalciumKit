import jax
import jax.numpy as jnp
import numpy as np

from CalciumKit.eta import simple_eta


def test_simple_eta_example_smoke():
    """
    Smoke test using the example from the doc:
    - Builds a noisy sinusoid
    - Runs simple_eta with dt=0.01
    - Verifies output shapes and basic sanity
    """
    T = 10_000
    t = jnp.linspace(0.0, 100.0, T)
    # deterministic key for test repeatability
    key = jax.random.PRNGKey(0)
    signal = jnp.sin(2 * jnp.pi * 0.5 * t) + 0.1 * jax.random.normal(key, (T,))
    events = jnp.array([10.0, 25.0, 40.0, 70.0, 90.0])

    avg, lags, traces = simple_eta(
        signal, t, window=(-1.0, 2.0), events=events, dt=0.01, return_traces=True
    )

    # Shapes
    assert avg.ndim == 1
    assert lags.ndim == 1
    assert traces.ndim == 2
    W = lags.shape[0]
    assert avg.shape == (W,)
    assert traces.shape == (events.shape[0], W)

    # Lags span window (inclusive)
    np.testing.assert_allclose(lags[0], -1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(lags[-1], 2.0, rtol=0, atol=1e-6)

    # Not all-NaN; at least some finite values produced
    assert jnp.isfinite(avg).any()


def test_simple_eta_recovers_template_exact():
    """
    Deterministic recovery: add the same template after several events on a uniform grid,
    aligned with the ETA sampling grid so the average equals the template (no edges).
    """
    # Uniform grid
    T = 5000
    t = jnp.linspace(0.0, 50.0, T)
    dt = float(t[1] - t[0])

    # Build a template of length W (inclusive endpoints)
    W = 101
    lags = jnp.linspace(0.0, (W - 1) * dt, W)
    template = jnp.exp(-lags / (lags[-1] + 1e-12))  # some smooth positive bump

    # Empty signal, then stamp template starting at each event
    signal = jnp.zeros((T,), dtype=jnp.float32)

    # Choose event indices so windows are fully in-bounds
    start_idx = 800
    gap = 700
    idxs = jnp.array([start_idx, start_idx + gap, start_idx + 2 * gap])
    events = t[idxs]  # exact times on grid

    # Add template to the signal at each event index
    # (window is [0, (W-1)*dt], exactly aligned to grid)
    for i in idxs:
        sl = slice(int(i), int(i) + W)
        signal = signal.at[sl].add(template)

    # Run ETA with matching window/grid (num=W so we sample exactly on the grid)
    avg, lags_out, traces = simple_eta(
        signal, t, window=(0.0, (W - 1) * dt), events=events, num=W, return_traces=True
    )

    # Shapes correct
    assert avg.shape == (W,)
    assert lags_out.shape == (W,)
    assert traces.shape == (events.shape[0], W)

    # Recover the template (exact up to float error)
    np.testing.assert_allclose(avg, template, rtol=0, atol=1e-5)
    # Each per-event trace equals the template
    np.testing.assert_allclose(traces, jnp.tile(template[None, :], (events.shape[0], 1)), rtol=0, atol=1e-5)


def test_simple_eta_nan_edges_and_counts():
    """
    Edge handling: include an event near the end so the window exceeds data.
    Expect NaNs in the out-of-bounds region of the traces; avg should be finite for
    in-bounds lags where at least one event contributes.
    """
    T = 2000
    t = jnp.linspace(0.0, 20.0, T)
    dt = float(t[1] - t[0])

    # Simple deterministic signal
    signal = jnp.sin(2 * jnp.pi * 0.2 * t)

    # Window of ~1 second with fine grid
    window = (-0.5, 0.5)
    num = 101  # dense grid
    W = num

    # One safe event in the middle, one near the end to cause OOB
    mid_idx = 1000
    near_end_idx = T - int(0.25 / dt)  # last quarter-second is tight
    events = jnp.array([t[mid_idx], t[near_end_idx]])

    avg, lags, traces = simple_eta(signal, t, window=window, events=events, num=num, return_traces=True)

    assert traces.shape == (2, W)
    second = traces[1]
    assert jnp.isnan(second).any()  # still true

    # Define an inner band that is safe for *both* events
    margins = jnp.minimum(events - t[0], t[-1] - events)  # per-event safe half-window
    inner_half = float(jnp.min(margins)) - 1e-9
    inner = (lags >= -inner_half) & (lags <= inner_half)

    assert inner.any(), "Inner region should not be empty"
    assert jnp.isfinite(traces[:, inner]).all()
