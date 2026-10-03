"""Independent masked corruption, eq 1."""

import jax
import jax.numpy as jnp


def alpha_bar(t):
    """Linear keep probability. alpha_bar(0) = 1 and alpha_bar(1) = 0."""
    return 1.0 - t


def corrupt(key, tokens, alpha_bar_t, mask_id):
    """Replace each token by mask_id with probability 1 - alpha_bar_t.

    qt(z^i | x^i) = alpha_bar_t * 1{z^i = x^i} + (1 - alpha_bar_t) * 1{z^i = m}.
    The keep decision is independent across positions. alpha_bar_t may be a
    scalar or any array that broadcasts to tokens.
    """
    tokens = jnp.asarray(tokens)
    keep = jax.random.bernoulli(key, alpha_bar_t, shape=tokens.shape)
    mask_token = jnp.asarray(mask_id, dtype=tokens.dtype)
    return jnp.where(keep, tokens, mask_token)


def sample_masked(key, tokens, mask_id, t_min=0.0, t_max=1.0):
    """Draw one time per sequence, then corrupt each position independently.

    Returns corrupted tokens, a boolean mask (True where the token is masked),
    and the sampled times. Times are uniform on [t_min, t_max).
    """
    tokens = jnp.asarray(tokens)
    key_t, key_m = jax.random.split(key)
    if tokens.ndim == 1:
        t = jax.random.uniform(key_t, shape=(), minval=t_min, maxval=t_max)
    else:
        t = jax.random.uniform(
            key_t, shape=(tokens.shape[0], 1), minval=t_min, maxval=t_max
        )
    z = corrupt(key_m, tokens, alpha_bar(t), mask_id)
    return z, z == mask_id, jnp.squeeze(t, axis=-1) if t.ndim else t
