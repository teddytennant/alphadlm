"""Fixed-NFE confidence sampler.

At each step the denoiser scores every still-masked position. The sampler
reveals a fixed count of the most confident positions (highest maximum
token probability) and writes their argmax tokens. The rest stay masked.
One model call is made per step, including steps with nothing left to
reveal. This is parallel confidence unmasking, not ancestral token-by-token
sampling.
"""

import jax.numpy as jnp


def confidence_sample(predict_fn, tokens, mask_id, n_steps, reveal_per_step):
    """Reveal top-confidence masked tokens for a fixed number of model calls.

    predict_fn(tokens) -> probabilities with shape (L, V) over the clean
    vocabulary. tokens is an int array of shape (L,). mask_id is the mask
    token id and must lie outside 0..V-1. Ties in confidence break toward
    the lower index.

    Returns the updated token array and the number of predict_fn calls,
    which equals n_steps.
    """
    tokens = jnp.asarray(tokens)
    n_calls = 0
    reveal_per_step = int(reveal_per_step)
    for _ in range(int(n_steps)):
        probs = jnp.asarray(predict_fn(tokens))
        n_calls += 1
        mask = tokens == mask_id
        n_masked = int(jnp.sum(mask))
        if n_masked == 0 or reveal_per_step <= 0:
            continue
        k = min(reveal_per_step, n_masked)
        confidence = jnp.max(probs, axis=-1)
        confidence = jnp.where(mask, confidence, -jnp.inf)
        index = jnp.arange(tokens.shape[0])
        # Primary key is -confidence so the largest confidence sorts first.
        order = jnp.lexsort((index, -confidence))
        chosen = order[:k]
        pred = jnp.argmax(probs, axis=-1)
        tokens = tokens.at[chosen].set(pred[chosen])
    return tokens, n_calls
