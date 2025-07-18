import jax.numpy as jnp
from CalciumKit.kalman import kalman_filter_smoother, kalman_filter_scan, rts_smoother_scan


def test_kalman_filter_identity_system():
    """
    Identity system with small process and measurement noise.
    Expected to track a constant signal well.
    """
    T = 10
    A = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    Q = jnp.array([[1e-4]])
    R = jnp.array([[1e-2]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])
    y = jnp.ones((T, 1))

    smoothed_means, smoothed_covs = kalman_filter_smoother(y, A, C, Q, R, m0, P0)

    assert smoothed_means.shape == (T, 1)
    assert smoothed_covs.shape == (T, 1, 1)
    assert jnp.allclose(smoothed_means[-1], 1.0, atol=0.1)


def test_kalman_filter_zero_measurement_noise():
    """
    With zero measurement noise, the filter should trust observations completely.
    """
    T = 5
    A = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    Q = jnp.array([[1e-3]])
    R = jnp.array([[0.0]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])
    y = jnp.linspace(0, 1, T).reshape(T, 1)

    means, covs = kalman_filter_scan(y, A, C, Q, R, m0, P0)

    assert jnp.allclose(means.squeeze(), y.squeeze(), atol=1e-4)


def test_kalman_filter_zero_process_noise():
    """
    With zero process noise, system assumes no dynamics uncertainty.
    """
    T = 5
    A = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    Q = jnp.array([[0.0]])
    R = jnp.array([[1e-2]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])
    y = jnp.linspace(0, 4, T).reshape(T, 1)

    means, _ = kalman_filter_scan(y, A, C, Q, R, m0, P0)
    # Not perfect tracking, but shape should be correct and values finite
    assert means.shape == (T, 1)
    assert jnp.all(jnp.isfinite(means))


def test_rts_smoother_monotonicity():
    """
    The smoothed estimate should be at least as accurate as the filtered estimate.
    That is, its final variance should be less than or equal to the filtered one.
    """
    T = 20
    A = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    Q = jnp.array([[1e-2]])
    R = jnp.array([[1e-1]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])
    y = jnp.sin(jnp.linspace(0, 3.14, T)).reshape(T, 1)

    filtered_means, filtered_covs = kalman_filter_scan(y, A, C, Q, R, m0, P0)
    smoothed_means, smoothed_covs = rts_smoother_scan(filtered_means, filtered_covs, A, Q)

    # Smoothed estimates should be closer to the middle of the distribution
    assert smoothed_means.shape == (T, 1)
    assert smoothed_covs.shape == (T, 1, 1)

    # Final smoothed covariance <= filtered covariance
    assert (smoothed_covs[-1] <= filtered_covs[-1]).all()


def test_long_sequence_stability():
    """
    Ensure numerical stability and runtime performance on longer sequences.
    """
    T = 1000
    A = jnp.array([[0.95]])
    C = jnp.array([[1.0]])
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])
    y = jnp.cumsum(jnp.ones(T)).reshape(T, 1)  # linearly increasing signal

    smoothed_means, smoothed_covs = kalman_filter_smoother(y, A, C, Q, R, m0, P0)

    # Check that output is valid and of expected size
    assert smoothed_means.shape == (T, 1)
    assert smoothed_covs.shape == (T, 1, 1)
    assert jnp.all(jnp.isfinite(smoothed_means))
    assert jnp.all(jnp.isfinite(smoothed_covs))