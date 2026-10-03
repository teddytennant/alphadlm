"""Tests for the sequence-level alpha loss and its population optimum."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from alphadlm.corrupt import alpha_bar, corrupt
from alphadlm.losses import (
    alpha_loss,
    ell_k,
    mean_token_cross_entropy,
    sequence_alpha_loss,
    tokenwise_alpha_loss,
)
from alphadlm.optimum import (
    expected_sequence_loss,
    figure2_joint,
    lower_temperature_target,
    marginals,
    minimize_factorized_sequence,
    minimize_factorized_tokenwise,
    minimize_j,
)
from alphadlm.sampler import confidence_sample
from alphadlm.train import train


def test_alpha_loss_limit_matches_neg_log():
    probs = jnp.array([0.15, 0.4, 0.7, 0.92])
    alpha = 1e-6
    estimate = alpha_loss(probs, alpha)
    reference = -jnp.log(probs)
    assert jnp.allclose(estimate, reference, atol=1e-4, rtol=0.0)
    # alpha = 1 is 1 - p(y), a separate check of eq 5.
    assert jnp.allclose(alpha_loss(probs, 1.0), 1.0 - probs, atol=1e-12)


def test_sequence_alpha_one_equals_one_minus_prod():
    token_probs = jnp.array([0.2, 0.5, 0.8])
    got = sequence_alpha_loss(token_probs, 1.0)
    assert jnp.allclose(got, 1.0 - jnp.prod(token_probs), atol=1e-12)


def test_sequence_alpha_not_sum_of_tokenwise():
    token_probs = jnp.array([0.4, 0.7, 0.2])
    seq = sequence_alpha_loss(token_probs, 1.0)
    tok = tokenwise_alpha_loss(token_probs, 1.0)
    assert jnp.allclose(seq, 1.0 - jnp.prod(token_probs), atol=1e-12)
    assert not jnp.allclose(seq, tok, atol=1e-6)
    assert float(tok) > float(seq)


def test_ell_k_limit_and_gradient_weight():
    logits = jax.random.normal(jax.random.PRNGKey(0), (4, 5))
    targets = jnp.array([1, 0, 3, 2])
    mask = jnp.array([True, True, False, True])

    def mean_ce(weights):
        log_probs = jax.nn.log_softmax(weights, axis=-1)
        return mean_token_cross_entropy(log_probs, targets, mask)

    def loss_k(weights, k):
        log_probs = jax.nn.log_softmax(weights, axis=-1)
        return ell_k(log_probs, targets, mask, k)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_lp = log_probs[jnp.arange(4), targets]
    s = jnp.mean(token_lp[mask])

    # Limit k -> 0, evaluated with the expm1 formula, not the k = 0 branch.
    small_k = 1e-6
    assert abs(small_k) > 1e-12
    assert jnp.allclose(loss_k(logits, small_k), -s, atol=1e-4, rtol=0.0)
    assert jnp.allclose(mean_ce(logits), -s, atol=1e-12)

    k = 1.25
    value = loss_k(logits, k)
    manual = (1.0 - jnp.exp(k * s)) / k
    assert jnp.allclose(value, manual, atol=1e-10, rtol=1e-10)

    g_k = jax.grad(lambda weights: loss_k(weights, k))(logits)
    g_ce = jax.grad(mean_ce)(logits)
    scaled = jnp.exp(k * s) * g_ce
    assert jnp.allclose(g_k, scaled, rtol=1e-6, atol=1e-8)

    # A second k, still tight, and an empty mask contributes zero.
    k2 = 3.0
    g_k2 = jax.grad(lambda weights: loss_k(weights, k2))(logits)
    s2 = s
    assert jnp.allclose(g_k2, jnp.exp(k2 * s2) * g_ce, rtol=1e-6, atol=1e-8)
    empty = jnp.array([False, False, False, False])
    assert float(ell_k(log_probs, targets, empty, k)) == 0.0


def test_figure2_factorized_ce_ungrammatical_mass():
    q = figure2_joint()
    p1, p2 = marginals(q)
    joint = p1[:, None] * p2[None, :]
    invalid = q == 0
    mass = jnp.sum(joint * invalid)
    # (I, is) + (he, am) + (she, am) = 0.24 + 0.14 + 0.10.
    assert jnp.allclose(mass, 0.48, atol=1e-12)
    assert jnp.allclose(p1, jnp.array([0.40, 0.35, 0.25]), atol=1e-12)
    assert jnp.allclose(p2, jnp.array([0.40, 0.60]), atol=1e-12)


def test_theorem_endpoints_factorized():
    q = figure2_joint()
    q1, q2 = marginals(q)

    p1, p2, _loss = minimize_factorized_sequence(q, 0.0)
    assert jnp.allclose(p1, q1, atol=1e-6, rtol=0.0), p1
    assert jnp.allclose(p2, q2, atol=1e-6, rtol=0.0), p2

    p1, p2, loss = minimize_factorized_sequence(q, 1.0)
    assert jnp.allclose(p1, jnp.array([1.0, 0.0, 0.0]), atol=1e-8), p1
    assert jnp.allclose(p2, jnp.array([1.0, 0.0]), atol=1e-8), p2
    # Unique joint mode (I, am). Loss equals 1 - q(mode).
    assert jnp.allclose(loss, 1.0 - 0.40, atol=1e-8)

    t1, t2, _tloss = minimize_factorized_tokenwise(q, 1.0)
    # Token modes are I and is, so the product is the ungrammatical cell (I, is).
    assert jnp.allclose(t1, jnp.array([1.0, 0.0, 0.0]), atol=1e-8), t1
    assert jnp.allclose(t2, jnp.array([0.0, 1.0]), atol=1e-8), t2


def test_characterization_and_alpha_half_support():
    q = figure2_joint()
    alpha = 0.5
    p1, p2, loss = minimize_factorized_sequence(q, alpha)
    pi, _j_loss = minimize_j(q, alpha)
    m1, m2 = marginals(pi)
    assert jnp.allclose(p1, m1, atol=2e-2, rtol=2e-2), (p1, m1)
    assert jnp.allclose(p2, m2, atol=2e-2, rtol=2e-2), (p2, m2)

    joint = p1[:, None] * p2[None, :]
    # Paper claim at alpha = 0.5: support only on (he, is) and (she, is).
    invalid_and_mode = jnp.array(
        [
            [True, True],
            [True, False],
            [True, False],
        ]
    )
    leaked = jnp.sum(joint * invalid_and_mode)
    assert float(leaked) < 1e-3, joint
    assert float(joint[1, 1]) > 0.2, joint
    assert float(joint[2, 1]) > 0.2, joint
    # Group 2 beats the joint mode, so this is not the alpha = 1 endpoint.
    mode_loss = expected_sequence_loss(
        jnp.array([1.0, 0.0, 0.0]), jnp.array([1.0, 0.0]), q, alpha
    )
    assert float(loss) < float(mode_loss) - 1e-3


def test_lower_temperature_target():
    q = figure2_joint()
    assert jnp.allclose(lower_temperature_target(q, 0.0), q, atol=1e-12)
    qt = lower_temperature_target(q, 0.5)
    raw = jnp.array([0.40**2, 0.35**2, 0.25**2])
    raw = raw / jnp.sum(raw)
    assert jnp.allclose(qt[0, 0], raw[0], atol=1e-12)
    assert jnp.allclose(qt[1, 1], raw[1], atol=1e-12)
    assert jnp.allclose(qt[2, 1], raw[2], atol=1e-12)
    assert float(qt[0, 1]) == 0.0
    assert float(qt[1, 0]) == 0.0
    assert float(qt[2, 0]) == 0.0


def test_corruption_alpha_bar_endpoints():
    tokens = jnp.array([0, 3, 4])
    kept = corrupt(jax.random.PRNGKey(0), tokens, alpha_bar(0.0), 9)
    masked = corrupt(jax.random.PRNGKey(1), tokens, alpha_bar(1.0), 9)
    assert float(alpha_bar(0.0)) == 1.0
    assert float(alpha_bar(1.0)) == 0.0
    assert jnp.array_equal(kept, tokens)
    assert jnp.array_equal(masked, jnp.array([9, 9, 9]))


def _fixed_probs():
    # Confidences 0.20, 0.90, 0.50, 0.70, 0.40. Argmax tokens 1, 0, 3, 2, 1.
    conf = jnp.array([0.20, 0.90, 0.50, 0.70, 0.40])
    argmax = jnp.array([1, 0, 3, 2, 1])
    probs = jnp.zeros((5, 4))
    for i in range(5):
        probs = probs.at[i, argmax[i]].set(conf[i])
        rest = (1.0 - conf[i]) / 3.0
        for v in range(4):
            if v != int(argmax[i]):
                probs = probs.at[i, v].set(rest)
    return probs, argmax


def test_confidence_sampler_topk_and_call_count():
    probs, argmax = _fixed_probs()
    calls = {"n": 0}

    def predict_fn(tokens):
        calls["n"] += 1
        return probs

    mask_id = 7
    tokens = jnp.full((5,), mask_id)
    out, n_calls = confidence_sample(
        predict_fn, tokens, mask_id, n_steps=1, reveal_per_step=2
    )
    assert n_calls == 1
    assert calls["n"] == 1
    # Top two confidences are index 1 (0.90) and index 3 (0.70).
    assert int(out[1]) == int(argmax[1])
    assert int(out[3]) == int(argmax[3])
    assert int(out[0]) == mask_id
    assert int(out[2]) == mask_id
    assert int(out[4]) == mask_id

    calls["n"] = 0
    out3, n_calls3 = confidence_sample(
        predict_fn, tokens, mask_id, n_steps=3, reveal_per_step=1
    )
    assert n_calls3 == 3
    assert calls["n"] == 3
    # Next is index 2 (0.50). The two lowest stay masked.
    assert int(out3[1]) == int(argmax[1])
    assert int(out3[3]) == int(argmax[3])
    assert int(out3[2]) == int(argmax[2])
    assert int(out3[0]) == mask_id
    assert int(out3[4]) == mask_id


def test_confidence_sampler_reveals_in_parallel():
    # One step must use a single prediction for every revealed token.
    base = jnp.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    )
    flipped = jnp.array(
        [
            [0.1, 0.9],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )
    seen = []

    def predict_fn(tokens):
        seen.append(jnp.array(tokens))
        if jnp.any(tokens != 9):
            return flipped
        return base

    out, n_calls = confidence_sample(
        predict_fn, jnp.full((3,), 9), 9, n_steps=1, reveal_per_step=2
    )
    assert n_calls == 1
    assert len(seen) == 1
    # Top two by base confidence: index 0 (0.9) and index 1 (0.8), not index 2 (0.6).
    assert int(out[0]) == 0
    assert int(out[1]) == 1
    assert int(out[2]) == 9


def test_tiny_train_loss_decreases():
    result = train(steps=40, batch_size=64, eval_size=128, k=2.0, seed=0)
    assert result["final_loss"] < result["initial_loss"]
    assert result["final_loss"] == result["final_loss"]
