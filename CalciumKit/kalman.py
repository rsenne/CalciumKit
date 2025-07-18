import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve


def kalman_predict(m, P, A, Q):
    """
    Predict the next state and covariance using the state transition model.

    Args:
        m (jnp.ndarray): The current state mean.
        P (jnp.ndarray): The current state covariance.
        A (jnp.ndarray): The state transition matrix.
        Q (jnp.ndarray): The process noise covariance.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The predicted state mean and covariance.
    """
    m_pred = A @ m
    P_pred = A @ P @ A.T + Q
    return m_pred, P_pred

def kalman_update_joseph(m_pred, P_pred, y_t, C, R):
    """
    Update the state estimate and covariance using the Kalman gain and measurement.

    Args:
        m_pred (jnp.ndarray): The predicted state mean.
        P_pred (jnp.ndarray): The predicted state covariance.
        y_t (jnp.ndarray): The current measurement.
        C (jnp.ndarray): The measurement matrix.
        R (jnp.ndarray): The measurement noise covariance.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The updated state mean and covariance.
    """
    S = C @ P_pred @ C.T + R
    chol_S = cho_factor(S, lower=True)
    K = cho_solve(chol_S, C @ P_pred).T
    innovation = y_t - C @ m_pred
    m_post = m_pred + K @ innovation
    I = jnp.eye(P_pred.shape[0])
    I_KC = I - K @ C
    P_post = I_KC @ P_pred @ I_KC.T + K @ R @ K.T
    return m_post, P_post

def kalman_filter_scan(y, A, C, Q, R, m0, P0):
    """
    Perform the Kalman filter using JAX's scan for efficient computation.

    Args:
        y (jnp.ndarray): The measurements.
        A (jnp.ndarray): The state transition matrix.
        C (jnp.ndarray): The measurement matrix.
        Q (jnp.ndarray): The process noise covariance.
        R (jnp.ndarray): The measurement noise covariance.
        m0 (jnp.ndarray): The initial state mean.
        P0 (jnp.ndarray): The initial state covariance.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The filtered state means and covariances.
    """
    def step(carry, y_t):
        m, P = carry
        m_pred, P_pred = kalman_predict(m, P, A, Q)
        m_post, P_post = kalman_update_joseph(m_pred, P_pred, y_t, C, R)
        return (m_post, P_post), (m_post, P_post)

    init_carry = (m0, P0)
    (_, _), (means, covs) = lax.scan(step, init_carry, y)
    return means, covs

def rts_gain(P_filt, A, Q):
    """
    Compute the Rauch-Tung-Striebel (RTS) gain and predicted covariance.

    Args:
        P_filt (jnp.ndarray): The filtered state covariance.
        A (jnp.ndarray): The state transition matrix.
        Q (jnp.ndarray): The process noise covariance.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The RTS gain and predicted covariance.
    """
    P_pred = A @ P_filt @ A.T + Q
    chol_P_pred = cho_factor(P_pred, lower=True)
    G = cho_solve(chol_P_pred, A @ P_filt).T
    return G, P_pred

def rts_update_joseph(m_filt, P_filt, m_next, P_next, A, Q):
    """
    Perform the Rauch-Tung-Striebel (RTS) smoother update using Joseph form.

    Args:
        m_filt (jnp.ndarray): The filtered state mean.
        P_filt (jnp.ndarray): The filtered state covariance.
        m_next (jnp.ndarray): The next state mean.
        P_next (jnp.ndarray): The next state covariance.
        A (jnp.ndarray): The state transition matrix.
        Q (jnp.ndarray): The process noise covariance.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The smoothed state mean and covariance.
    """
    G, P_pred = rts_gain(P_filt, A, Q)
    m_smooth = m_filt + G @ (m_next - A @ m_filt)
    P_smooth = P_filt + G @ (P_next - P_pred) @ G.T
    return m_smooth, P_smooth

def rts_smoother_scan(filtered_means, filtered_covs, A, Q):
    """
    Perform the Rauch-Tung-Striebel (RTS) smoother using JAX's scan.

    Args:
        filtered_means (jnp.ndarray): The filtered state means.
        filtered_covs (jnp.ndarray): The filtered state covariances.
        A (jnp.ndarray): The state transition matrix.
        Q (jnp.ndarray): The process noise covariance.
    """
    def step(carry, inputs):
        m_next, P_next = carry
        m_filt, P_filt = inputs
        m_smooth, P_smooth = rts_update_joseph(m_filt, P_filt, m_next, P_next, A, Q)
        return (m_smooth, P_smooth), (m_smooth, P_smooth)

    init_carry = (filtered_means[-1], filtered_covs[-1])
    inputs = (filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    (_, _), (rev_means, rev_covs) = lax.scan(step, init_carry, inputs)

    smoothed_means = jnp.concatenate([rev_means[::-1], filtered_means[-1:]], axis=0)
    smoothed_covs = jnp.concatenate([rev_covs[::-1], filtered_covs[-1:]], axis=0)
    return smoothed_means, smoothed_covs

@jax.jit
def kalman_filter_smoother(y, A, C, Q, R, m0, P0):
    """
    Perform the Kalman filter and smoother.

    Args:
        y (jnp.ndarray): The measurements.
        A (jnp.ndarray): The state transition matrix.
        C (jnp.ndarray): The measurement matrix.
        Q (jnp.ndarray): The process noise covariance.
        R (jnp.ndarray): The measurement noise covariance.
        m0 (jnp.ndarray): The initial state mean.
        P0 (jnp.ndarray): The initial state covariance.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The smoothed state means and covariances.
    """
    filtered_means, filtered_covs = kalman_filter_scan(y, A, C, Q, R, m0, P0)
    smoothed_means, smoothed_covs = rts_smoother_scan(filtered_means, filtered_covs, A, Q)
    return smoothed_means, smoothed_covs