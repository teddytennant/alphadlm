"""Tiny factorized denoiser.

The network reads the full corrupted sequence and emits an independent
categorical distribution at each position. Factorization is in the output,
not in the features: later sampler steps can condition on tokens revealed
earlier, but one evaluation does not model a joint over tokens.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map


def init_params(key, vocab_size, length, dim=32, hidden=64):
    """vocab_size is the clean vocabulary. The embedding table also holds the mask."""
    k_emb, k_pos, k_w1, k_w2 = jax.random.split(key, 4)
    fan_in = length * dim
    params = {
        "embed": jax.random.normal(k_emb, (vocab_size + 1, dim)) * 0.02,
        "pos": jax.random.normal(k_pos, (length, dim)) * 0.02,
        "w1": jax.random.normal(k_w1, (fan_in, hidden)) * jnp.sqrt(2.0 / fan_in),
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(k_w2, (hidden, length * vocab_size))
        * jnp.sqrt(2.0 / hidden),
        "b2": jnp.zeros((length * vocab_size,)),
    }
    return params


def apply(params, tokens):
    """Logits over the clean vocabulary. tokens may be (L,) or (B, L)."""
    tokens = jnp.asarray(tokens)
    single = tokens.ndim == 1
    if single:
        tokens = tokens[None, :]
    length = params["pos"].shape[0]
    vocab = params["embed"].shape[0] - 1
    hidden_in = params["embed"][tokens] + params["pos"]
    flat = hidden_in.reshape((tokens.shape[0], length * hidden_in.shape[-1]))
    hidden = jax.nn.gelu(flat @ params["w1"] + params["b1"])
    logits = hidden @ params["w2"] + params["b2"]
    logits = logits.reshape((tokens.shape[0], length, vocab))
    if single:
        return logits[0]
    return logits


def predict_probs(params, tokens):
    return jax.nn.softmax(apply(params, tokens), axis=-1)


def adam_init(params):
    return {
        "m": tree_map(jnp.zeros_like, params),
        "v": tree_map(jnp.zeros_like, params),
        "t": jnp.array(0, dtype=jnp.int32),
    }


def adam_update(params, state, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, state["m"], grads)
    v = tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = tree_map(lambda mi: mi / (1.0 - b1**t), m)
    vhat = tree_map(lambda vi: vi / (1.0 - b2**t), v)
    new_params = tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return new_params, {"m": m, "v": v, "t": t}
