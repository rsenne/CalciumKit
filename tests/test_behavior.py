import jax.numpy as jnp
import pytest
from CalciumKit.behavior import (
    convert_degrees_to_positions,
    create_state_matrix,
    wheel_params,
    wheel_kinetics,
    RAMIREZ_WHEEL_RADIUS
)


def test_convert_degrees_simple():
    """Test basic degree to position conversion without wraparound."""
    degrees = jnp.array([0.0, 90.0, 180.0, 270.0, 360.0])
    positions = convert_degrees_to_positions(degrees)
    
    # Full rotation (360 degrees) should equal circumference
    circumference = 2 * jnp.pi * RAMIREZ_WHEEL_RADIUS
    assert jnp.allclose(positions[-1], circumference, rtol=1e-5)
    
    # Should be monotonically increasing
    assert jnp.all(jnp.diff(positions) >= 0)


def test_convert_degrees_wraparound():
    """Test wraparound handling at ±180 degrees."""
    # Simulate going from 170 to -170 (should be continuous)
    degrees = jnp.array([170.0, 175.0, 180.0, -175.0, -170.0])
    positions = convert_degrees_to_positions(degrees)
    
    # Should be monotonically increasing (no jumps)
    diffs = jnp.diff(positions)
    assert jnp.all(diffs > 0), "Position should increase smoothly through wraparound"
    assert jnp.all(diffs < 0.01), "No large jumps expected for small angle changes"


def test_convert_degrees_asymmetric_range():
    """Test handling of asymmetric range (-180, 180.1]."""
    # Simulate data that goes to 180.1 as part of normal convention
    degrees = jnp.array([178.0, 179.0, 180.0, 180.1, -179.9, -179.0])
    positions = convert_degrees_to_positions(degrees)
    
    # Should be continuous with no large jumps
    diffs = jnp.diff(positions)
    assert jnp.all(jnp.abs(diffs) < 0.001), "Should handle 180.1 smoothly without remapping"
    assert jnp.all(jnp.isfinite(positions)), "All positions should be finite"
    
    # Verify monotonic increase through the boundary
    assert jnp.all(diffs > 0), "Motion should be consistently forward"


def test_convert_degrees_negative_motion():
    """Test backward motion (negative degrees)."""
    degrees = jnp.array([0.0, -90.0, -180.0, -270.0])
    positions = convert_degrees_to_positions(degrees)
    
    # Should be monotonically decreasing
    assert jnp.all(jnp.diff(positions) <= 0)


def test_convert_degrees_custom_radius():
    """Test with custom wheel radius."""
    custom_radius = 0.05  # 5 cm
    degrees = jnp.array([0.0, 360.0])
    positions = convert_degrees_to_positions(degrees, wheel_radius=custom_radius)
    
    expected_circumference = 2 * jnp.pi * custom_radius
    assert jnp.allclose(positions[-1], expected_circumference, rtol=1e-5)


def test_create_state_matrix():
    """Test state transition matrix construction."""
    delta_t = 0.1
    A = create_state_matrix(delta_t)
    
    assert A.shape == (3, 3)
    # Check diagonal is [1, 1, 1]
    assert jnp.allclose(jnp.diag(A), jnp.ones(3))
    # Check specific elements
    assert jnp.allclose(A[0, 1], delta_t)
    assert jnp.allclose(A[0, 2], 0.5 * delta_t**2)
    assert jnp.allclose(A[1, 2], delta_t)


def test_wheel_params_initialization():
    """Test parameter initialization for Kalman filter."""
    positions = jnp.array([0.0, 0.1, 0.22, 0.35, 0.5])
    delta_t = 0.01
    
    initial_state, initial_cov, A, C, Q, R = wheel_params(positions, delta_t)
    
    # Check shapes
    assert initial_state.shape == (3,)
    assert initial_cov.shape == (3, 3)
    assert A.shape == (3, 3)
    assert C.shape == (1, 3)
    assert Q.shape == (3, 3)
    assert R.shape == (1, 1)
    
    # Check observation matrix observes only position
    assert jnp.allclose(C, jnp.array([[1.0, 0.0, 0.0]]))
    
    # Check covariances are positive definite
    assert jnp.all(jnp.linalg.eigvals(initial_cov) > 0)
    assert jnp.all(jnp.linalg.eigvals(Q) > 0)
    assert R[0, 0] > 0


def test_wheel_kinetics_constant_velocity():
    """Test Kalman smoothing on constant velocity motion."""
    # Simulate constant velocity: position = v*t
    delta_t = 0.01
    velocity = 0.05  # m/s
    t = jnp.arange(0, 1.0, delta_t)
    true_positions = velocity * t
    
    # Add small noise
    noisy_positions = true_positions + 0.0001 * jnp.sin(10 * t)
    
    smoothed_means, smoothed_covs = wheel_kinetics(noisy_positions, delta_t)
    
    # Check output shapes
    assert smoothed_means.shape == (len(noisy_positions), 3)
    assert smoothed_covs.shape == (len(noisy_positions), 3, 3)
    
    # Smoothed positions should be close to true positions
    smoothed_positions = smoothed_means[:, 0]
    assert jnp.allclose(smoothed_positions, true_positions, atol=0.01)
    
    # Velocity estimate should converge to true velocity
    smoothed_velocity = smoothed_means[-1, 1]
    assert jnp.allclose(smoothed_velocity, velocity, atol=0.01)


def test_wheel_kinetics_constant_acceleration():
    """Test Kalman smoothing on constant acceleration motion."""
    # Simulate constant acceleration: position = 0.5*a*t^2
    delta_t = 0.01
    accel = 0.1  # m/s^2
    t = jnp.arange(0, 1.0, delta_t)
    true_positions = 0.5 * accel * t**2
    
    # Add noise
    noisy_positions = true_positions + 0.0001 * jnp.random.normal(0, 1, len(t))
    
    smoothed_means, smoothed_covs = wheel_kinetics(noisy_positions, delta_t)
    
    # Check shapes
    assert smoothed_means.shape == (len(noisy_positions), 3)
    
    # Smoothed positions should be close to true
    smoothed_positions = smoothed_means[:, 0]
    assert jnp.corrcoef(smoothed_positions, true_positions)[0, 1] > 0.99
    
    # Acceleration should be detected
    smoothed_accel = smoothed_means[-1, 2]
    assert jnp.allclose(smoothed_accel, accel, atol=0.05)


def test_wheel_kinetics_from_degrees():
    """End-to-end test: degrees to positions to kinetics."""
    # Simulate wheel rotating at constant angular velocity
    angular_vel = 10.0  # degrees per step
    degrees = jnp.cumsum(jnp.ones(100) * angular_vel)
    
    # Handle wraparound
    degrees = jnp.mod(degrees + 180, 360) - 180
    
    # Convert to positions
    positions = convert_degrees_to_positions(degrees)
    
    # Apply Kalman smoothing
    delta_t = 0.01
    smoothed_means, smoothed_covs = wheel_kinetics(positions, delta_t)
    
    # Check that output is valid
    assert smoothed_means.shape == (len(positions), 3)
    assert jnp.all(jnp.isfinite(smoothed_means))
    assert jnp.all(jnp.isfinite(smoothed_covs))
    
    # Velocity should be approximately constant
    velocities = smoothed_means[:, 1]
    assert jnp.std(velocities[20:]) < 0.01  # After initial transient


def test_wheel_kinetics_warns_on_degrees():
    """Test that warning is raised if degrees are passed instead of meters."""
    # Create fake "degree" data (large values)
    fake_degrees = jnp.linspace(0, 100, 50)
    
    with pytest.warns(UserWarning, match="Input may be in degrees"):
        wheel_kinetics(fake_degrees, delta_t=0.01)