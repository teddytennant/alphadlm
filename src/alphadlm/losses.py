"""Alpha losses for a factorized masked diffusion denoiser.

Equations follow arXiv 2609.38066. Sequence-level alpha loss applies
generalized cross-entropy to the joint factorized probability. It is not
the sum of per-token alpha losses. The training loss ell_k is that
objective with alpha_eff = k / m, scaled by 1/m, evaluated in log space.
"""

import jax.numpy as jnp


def alpha_loss(prob, alpha):
    """Generalized cross-entropy, eq 5.

    ell_alpha(p, y) = (1 - p(y)^alpha) / alpha for alpha > 0.
    The alpha -> 0 limit is -log p(y). Computed as -expm1(alpha * log p) / alpha
    so a small alpha does not cancel in float arithmetic.
    """
    prob = jnp.asarray(prob)
    alpha = jnp.asarray(alpha, dtype=prob.dtype)
    log_p = jnp.log(prob)
    stable = -jnp.expm1(alpha * log_p) / alpha
    limit = -log_p
    return jnp.where(jnp.abs(alpha) < 1e-12, limit, stable)


def sequence_alpha_loss(token_probs, alpha):
    """Sequence-level alpha loss, eq 6.

    token_probs holds p_i(y_i) on the last axis. The loss is
    (1 - prod_i p_i(y_i)^alpha) / alpha, not the sum of token losses.
    As alpha -> 0 this becomes the sum of token cross-entropies.
    """
    token_probs = jnp.asarray(token_probs)
    alpha = jnp.asarray(alpha, dtype=token_probs.dtype)
    sum_log = jnp.sum(jnp.log(token_probs), axis=-1)
    stable = -jnp.expm1(alpha * sum_log) / alpha
    limit = -sum_log
    return jnp.where(jnp.abs(alpha) < 1e-12, limit, stable)


def tokenwise_alpha_loss(token_probs, alpha):
    """Token-wise alpha loss, eq 11: sum of per-token eq-5 losses."""
    return jnp.sum(alpha_loss(token_probs, alpha), axis=-1)


def _token_log_probs(log_probs, targets):
    gathered = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    return jnp.squeeze(gathered, axis=-1)


def token_cross_entropy(log_probs, targets, mask=None):
    """Sum of token negative log probs, eq 3. Optional mask over positions."""
    nll = -_token_log_probs(log_probs, targets)
    if mask is None:
        return jnp.sum(nll, axis=-1)
    mask_f = mask.astype(nll.dtype)
    return jnp.sum(nll * mask_f, axis=-1)


def _mean_log_prob(log_probs, targets, mask):
    token_lp = _token_log_probs(log_probs, targets)
    mask_f = mask.astype(token_lp.dtype)
    m = jnp.sum(mask_f, axis=-1)
    sum_lp = jnp.sum(token_lp * mask_f, axis=-1)
    s = sum_lp / jnp.maximum(m, 1.0)
    return s, m


def mean_token_cross_entropy(log_probs, targets, mask):
    """Mean token cross-entropy -s over masked positions. Zero if none are masked."""
    s, m = _mean_log_prob(log_probs, targets, mask)
    return jnp.where(m > 0, -s, jnp.zeros_like(s))


def ell_k(log_probs, targets, mask, k):
    """Mask-normalized sequence-level loss, eq 13.

    s = (1/m) * sum_{i masked} log p_i(x^i)
    ell_k(s) = (1 - exp(k * s)) / k = -expm1(k * s) / k

    As k -> 0 the value is the mean token cross-entropy -s.
    For k > 0, d(ell_k)/d(theta) = exp(k * s) * d(-s)/d(theta).
    Positions that are not masked do not enter s. Examples with m = 0
    contribute zero. The product of probabilities is never formed, only
    the sum of log probabilities, so the loss does not underflow.
    """
    s, m = _mean_log_prob(log_probs, targets, mask)
    k_arr = jnp.asarray(k, dtype=s.dtype)
    stable = -jnp.expm1(k_arr * s) / k_arr
    limit = -s
    raw = jnp.where(jnp.abs(k_arr) < 1e-12, limit, stable)
    return jnp.where(m > 0, raw, jnp.zeros_like(raw))
