"""Tiny CPU train on the Figure 2 grammar with ell_k.

Sequences are length 2, drawn from the joint
(I, am)=0.40, (he, is)=0.35, (she, is)=0.25.
Corruption is independent masking. The denoiser is a small MLP.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from alphadlm.corrupt import sample_masked
from alphadlm.losses import ell_k
from alphadlm.model import adam_init, adam_update, apply, init_params

CLEAN_VOCAB = 5
MASK_ID = 5
LENGTH = 2
# Token ids: 0 I, 1 he, 2 she, 3 am, 4 is.
_PAIRS = jnp.array(
    [
        [0, 3],
        [1, 4],
        [2, 4],
    ],
    dtype=jnp.int32,
)
_PAIR_PROBS = jnp.array([0.40, 0.35, 0.25])


def sample_clean(key, batch):
    idx = jax.random.choice(key, 3, shape=(batch,), p=_PAIR_PROBS)
    return _PAIRS[idx]


def loss_on_batch(params, corrupted, clean, mask, k):
    logits = apply(params, corrupted)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.mean(ell_k(log_probs, clean, mask, k))


@jax.jit
def train_step(params, opt_state, clean, key, k, lr):
    key_c, _key_unused = jax.random.split(key)
    corrupted, mask, _t = sample_masked(key_c, clean, MASK_ID)

    def loss_fn(p):
        return loss_on_batch(p, corrupted, clean, mask, k)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, opt_state = adam_update(params, opt_state, grads, lr)
    return params, opt_state, loss


def train(
    steps=60,
    batch_size=64,
    eval_size=128,
    k=2.0,
    seed=0,
    lr=0.03,
    dim=32,
    hidden=64,
):
    """Train on synthetic masked sequences. Returns initial and final eval loss.

    The eval batch is a fixed corruption of a fixed clean sample, so the
    comparison is not confounded by a new mask draw.
    """
    key = jax.random.PRNGKey(seed)
    key, k_init, k_eval_x, k_eval_z = jax.random.split(key, 4)
    params = init_params(k_init, CLEAN_VOCAB, LENGTH, dim=dim, hidden=hidden)
    opt_state = adam_init(params)
    eval_clean = sample_clean(k_eval_x, eval_size)
    eval_z, eval_mask, _t = sample_masked(k_eval_z, eval_clean, MASK_ID)
    initial = loss_on_batch(params, eval_z, eval_clean, eval_mask, k)
    for _step in range(steps):
        key, k_data, k_step = jax.random.split(key, 3)
        clean = sample_clean(k_data, batch_size)
        params, opt_state, _loss = train_step(
            params, opt_state, clean, k_step, k, lr
        )
    final = loss_on_batch(params, eval_z, eval_clean, eval_mask, k)
    return {
        "initial_loss": float(initial),
        "final_loss": float(final),
        "steps": steps,
        "k": k,
        "seed": seed,
        "params": params,
    }


def main():
    result = train()
    print(f"steps={result['steps']} k={result['k']} seed={result['seed']}")
    print(f"initial_loss={result['initial_loss']:.8f}")
    print(f"final_loss={result['final_loss']:.8f}")


if __name__ == "__main__":
    main()
