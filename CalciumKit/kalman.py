import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve
from typing import Tuple

@jax.jit
def kalman_predict(m: jnp.ndarray, P: jnp.ndarray, A: jnp.ndarray, Q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Predict the next state and covariance using the state transition model.

    Args:
        m: The current state mean.
        P: The current state covariance.
        A: The state transition matrix.
        Q: The process noise covariance.

    Returns:
        The predicted state mean and covariance.
    """
    m_pred = A @ m
    P_pred = A @ P @ A.T + Q
    return m_pred, P_pred

@jax.jit
def kalman_update_joseph(
    m_pred: jnp.ndarray, P_pred: jnp.ndarray, y_t: jnp.ndarray, C: jnp.ndarray, R: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update the state estimate and covariance using the Joseph form for numerical stability.

    Args:
        m_pred: The predicted state mean.
        P_pred: The predicted state covariance.
        y_t: The current observation.
        C: The observation matrix.
        R: The observation noise covariance.

    Returns:
        The updated state mean and covariance.
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

@jax.jit
def kalman_filter_scan(
    y: jnp.ndarray, A: jnp.ndarray, C: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray, m0: jnp.ndarray, P0: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Kalman filtering over a sequence of observations using lax.scan.

    Args:
        y: Array of shape (T, obs_dim), the observations.
        A: The state transition matrix.
        C: The observation matrix.
        Q: The process noise covariance.
        R: The observation noise covariance.
        m0: Initial mean of the latent state.
        P0: Initial covariance of the latent state.

    Returns:
        The filtered means and covariances for each timestep.
    """
    def step(carry, y_t):
        m, P = carry
        m_pred, P_pred = kalman_predict(m, P, A, Q)
        m_post, P_post = kalman_update_joseph(m_pred, P_pred, y_t, C, R)
        return (m_post, P_post), (m_post, P_post)

    init_carry = (m0, P0)
    (_, _), (means, covs) = lax.scan(step, init_carry, y)
    
    return means, covs

@jax.jit
def rts_update_joseph(
    m_filt: jnp.ndarray,
    P_filt: jnp.ndarray,
    m_next: jnp.ndarray,
    P_next: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    One step of the RTS smoother using the Joseph update for numerical stability.

    Args:
        m_filt: Filtered mean at time t.
        P_filt: Filtered covariance at time t.
        m_next: Smoothed mean at time t+1.
        P_next: Smoothed covariance at time t+1.
        A: State transition matrix.
        Q: Process noise covariance.

    Returns:
        The smoothed mean and covariance at time t, and cross covariance E[x_t x_{t+1}^T].
    """
    P_pred = A @ P_filt @ A.T + Q
    chol_P_pred = cho_factor(P_pred, lower=True)
    G = cho_solve(chol_P_pred, A @ P_filt).T
    m_smooth = m_filt + G @ (m_next - A @ m_filt)
    P_smooth = P_filt + G @ (P_next - P_pred) @ G.T
    cross_cov = G @ P_next
    return m_smooth, P_smooth, cross_cov

@jax.jit
def rts_smoother_scan(
    filtered_means: jnp.ndarray, filtered_covs: jnp.ndarray, A: jnp.ndarray, Q: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run the Rauch-Tung-Striebel smoother using lax.scan in reverse.

    Args:
        filtered_means: The filtered state means from Kalman filter.
        filtered_covs: The filtered state covariances.
        A: The state transition matrix.
        Q: The process noise covariance.

    Returns:
        The smoothed state means, covariances, and cross-covariances.
    """
    
    def step(carry, inputs):
        m_next, P_next = carry
        m_filt, P_filt = inputs
        m_smooth, P_smooth, cross_cov = rts_update_joseph(m_filt, P_filt, m_next, P_next, A, Q)
        return (m_smooth, P_smooth), (m_smooth, P_smooth, cross_cov)

    init = (filtered_means[-1], filtered_covs[-1])
    inputs = (filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    (_, _), (rev_means, rev_covs, rev_cross) = lax.scan(step, init, inputs)

    smoothed_means = jnp.concatenate([rev_means[::-1], filtered_means[-1:]], axis=0)
    smoothed_covs = jnp.concatenate([rev_covs[::-1], filtered_covs[-1:]], axis=0)
    cross_covs = rev_cross[::-1]

    return smoothed_means, smoothed_covs, cross_covs

@jax.jit
def em_update_Q_R(
    y: jnp.ndarray,
    smoothed_means: jnp.ndarray,
    smoothed_covs: jnp.ndarray,
    cross_covs: jnp.ndarray,
    A: jnp.ndarray,
    C: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform the M-step of the EM algorithm to update Q and R.

    Args:
        y: Observations of shape (T, obs_dim).
        smoothed_means: Smoothed state means.
        smoothed_covs: Smoothed state covariances.
        cross_covs: Cross-covariances E[x_t x_{t-1}^T].
        A: The state transition matrix.
        C: The observation matrix.

    Returns:
        Updated estimates of Q and R.
    """
    T = y.shape[0]
    
    E_xt_xtT = jnp.einsum("ti,tj->tij", smoothed_means, smoothed_means) + smoothed_covs
    E_xtm1_xtT = cross_covs + jnp.einsum("ti,tj->tij", smoothed_means[1:], smoothed_means[:-1])

    def q_body(t, Q_sum):
        Q_sum += (
            E_xt_xtT[t]
            - A @ E_xtm1_xtT[t - 1].T
            - E_xtm1_xtT[t - 1] @ A.T
            + A @ E_xt_xtT[t - 1] @ A.T
        )
        return Q_sum

    Q_sum = lax.fori_loop(1, T, q_body, jnp.zeros_like(E_xt_xtT[0]))
    Q_new = Q_sum / (T - 1)

    def r_body(t, R_sum):
        err = y[t] - C @ smoothed_means[t]
        R_sum += err @ err.T + C @ smoothed_covs[t] @ C.T
        return R_sum

    R_sum = lax.fori_loop(0, T, r_body, jnp.zeros((C.shape[0], C.shape[0])))
    R_new = R_sum / T

    return Q_new, R_new

def kalman_em(
    y: jnp.ndarray,
    A: jnp.ndarray,
    C: jnp.ndarray,
    Q_init: jnp.ndarray,
    R_init: jnp.ndarray,
    m0: jnp.ndarray,
    P0: jnp.ndarray,
    num_iters: int = 10,
    tol: float = 1e-5,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run EM algorithm to estimate Q, R using Kalman filtering and RTS smoothing.

    Args:
        y: Observations of shape (T, obs_dim).
        A: State transition matrix.
        C: Observation matrix.
        Q_init: Initial estimate of Q.
        R_init: Initial estimate of R.
        m0: Initial state mean.
        P0: Initial state covariance.
        num_iters: Maximum number of EM iterations.
        tol: Convergence tolerance on log-likelihood proxy.
        verbose: Whether to print progress.

    Returns:
        Estimated Q, R, and smoothed state means and covariances.
    """
    Q, R = Q_init, R_init
    prev_ll = -jnp.inf

    for i in range(num_iters):
        filtered_means, filtered_covs = kalman_filter_scan(y, A, C, Q, R, m0, P0)
        smoothed_means, smoothed_covs, cross_covs = rts_smoother_scan(filtered_means, filtered_covs, A, Q)
        Q, R = em_update_Q_R(y, smoothed_means, smoothed_covs, cross_covs, A, C)

        ll_proxy = -jnp.sum(jnp.square(y - smoothed_means @ C.T))
        if verbose:
            print(f"Iter {i+1}, Proxy LL: {ll_proxy:.4f}")

        if jnp.abs(ll_proxy - prev_ll) < tol:
            break
        prev_ll = ll_proxy

    return Q, R, smoothed_means, smoothed_covs


@jax.jit
def kalman_filter_smoother(
    y: jnp.ndarray,
    A: jnp.ndarray,
    C: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    m0: jnp.ndarray,
    P0: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Kalman filter and RTS smoother in one pass.

    Args:
        y: Observations of shape (T, obs_dim).
        A: State transition matrix.
        C: Observation matrix.
        Q: Process noise covariance.
        R: Observation noise covariance.
        m0: Initial mean.
        P0: Initial covariance.

    Returns:
        Smoothed means and covariances for the entire sequence.
    """
    filtered_means, filtered_covs = kalman_filter_scan(y, A, C, Q, R, m0, P0)
    smoothed_means, smoothed_covs, _ = rts_smoother_scan(filtered_means, filtered_covs, A, Q)
    return smoothed_means, smoothed_covs
