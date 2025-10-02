import warnings
from typing import Tuple
import jax.numpy as jnp
import CalciumKit.kalman as kalman

RAMIREZ_WHEEL_RADIUS = 0.00875  # meters -> 8.75 cm

def convert_degrees_to_positions(degrees: jnp.ndarray, wheel_radius: float = RAMIREZ_WHEEL_RADIUS) -> jnp.ndarray:
    """
    Convert wheel rotation in degrees to cumulative linear displacement in meters.
    Handles wrap-around discontinuities (e.g., -180/180 boundary).

    Args:
        degrees: The degrees of wheel rotation over time. Typically in range (-180, 180.1].
        wheel_radius: The radius of the wheel in meters.

    Returns:
        The estimated position in meters over time.
    """
    # Unwrap handles discontinuities automatically
    unwrapped = jnp.unwrap(degrees * jnp.pi / 180.0)
    
    # Convert to linear displacement
    positions = (unwrapped / (2 * jnp.pi)) * (2 * jnp.pi * wheel_radius)
    
    return positions


def create_state_matrix(delta_t: float) -> jnp.ndarray:
    """
    Create the state transition matrix A for a constant-acceleration model.

    Returns:
        The (3x3) state transition matrix.
    """
    return jnp.array([
        [1.0, delta_t, 0.5 * delta_t**2],
        [0.0, 1.0,     delta_t],
        [0.0, 0.0,     1.0]
    ])


def wheel_params(position_vector: jnp.ndarray, delta_t: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Initialize state and noise parameters for wheel motion inference.

    Args:
        position_vector: The observed positions in meters.
        delta_t: Time step between samples.

    Returns:
        Tuple of (initial_state, initial_covariance, A, C, Q, R).
    """
    initial_position = position_vector[0]
    initial_velocity = (position_vector[1] - position_vector[0]) / delta_t
    initial_acceleration = (position_vector[2] - 2 * position_vector[1] + position_vector[0]) / (delta_t ** 2)

    initial_state = jnp.array([initial_position, initial_velocity, initial_acceleration])
    initial_covariance = jnp.diag(jnp.array([1e-4, 1e-2, 1e-1]))

    A = create_state_matrix(delta_t)

    Q = 0.04 * jnp.array([
        [0.25 * delta_t**4, 0.5 * delta_t**3, 0.5 * delta_t**2],
        [0.5 * delta_t**3,     delta_t**2,       delta_t],
        [0.5 * delta_t**2,     delta_t,          1.0]
    ])

    C = jnp.array([[1.0, 0.0, 0.0]])  # Observe only position
    R = jnp.array([[1e-8]])          # Low measurement noise

    return initial_state, initial_covariance, A, C, Q, R


def wheel_kinematics(position_vector: jnp.ndarray, delta_t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimate position, velocity, and acceleration of the wheel using Kalman smoothing.

    Args:
        position_vector: Observed wheel positions (meters), shape (T,).
        delta_t: Sampling interval (seconds).

    Returns:
        smoothed_means: State estimates at each time step, shape (T, 3)
        smoothed_covs: Covariance estimates at each time step, shape (T, 3, 3)
    """
    if jnp.max(jnp.abs(position_vector)) > 10.0:
        warnings.warn("Input may be in degrees. Did you forget to convert to meters?")

    # Reshape to (T, 1) for Kalman filter
    y = position_vector.reshape(-1, 1)
    
    initial_state, initial_covariance, A, C, Q, R = wheel_params(position_vector, delta_t)

    # run EM for the covariance parameters
    Q, R, _, _ = kalman.kalman_em(
        y=y,
        A=A,
        C=C,
        Q_init=Q,
        R_init=R,
        m0=initial_state,
        P0=initial_covariance
    )

    smoothed_means, smoothed_covs = kalman.kalman_filter_smoother(
        y=y,
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=initial_state,
        P0=initial_covariance
    )
    return smoothed_means, smoothed_covs