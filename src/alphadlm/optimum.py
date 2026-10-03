"""Population optimum of sequence-level alpha loss.

The factorized denoiser cannot represent every joint. Appendix B shows that
its sequence-level optimum is the marginals of a joint pi minimizing

    J(pi) = KL(pi || q) + alpha * sum_i H(pi_i).

This module minimizes that objective, and the factorized population loss,
numerically. It does not return a hardcoded Figure 2 table. Candidates are
simplex vertices, a product grid, a fine grid on each Cartesian group of
the support, and a short Adam polish. The lowest evaluated loss wins.

Conventions from the paper: 0 log 0 = 0, and KL is infinite when pi puts
mass outside the support of q. At alpha = 0 the population sequence loss is
the cross-entropy limit, expected sum of token negative log probabilities.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map


FIGURE2_POS1 = ("I", "he", "she")
FIGURE2_POS2 = ("am", "is")

# 20 hits the Figure 2 marginals (0.40, 0.35, 0.25) and (0.40, 0.60) exactly.
_GRID_RESOLUTION = 20
_GROUP_RESOLUTION = 200
_POLISH_STEPS = 400
_POLISH_LR = 0.15


def figure2_joint():
    """Target q from Figure 2. Axis 0 is (I, he, she), axis 1 is (am, is)."""
    q = jnp.zeros((3, 2))
    q = q.at[0, 0].set(0.40)
    q = q.at[1, 1].set(0.35)
    q = q.at[2, 1].set(0.25)
    return q


def marginals(joint):
    """Token marginals of a length-2 joint."""
    joint = jnp.asarray(joint)
    return joint.sum(axis=1), joint.sum(axis=0)


def lower_temperature_target(q, alpha):
    """Eq 8: q_T(y) = q(y)^(1/T) / sum q^(1/T), T = 1 - alpha.

    Entries with q(y) = 0 stay zero. alpha = 0 gives T = 1 and returns q
    when q is already normalized.
    """
    q = jnp.asarray(q)
    t = 1.0 - alpha
    log_q = jnp.where(q > 0, jnp.log(q), -jnp.inf)
    flat = jax.nn.softmax((log_q / t).reshape(-1))
    return flat.reshape(q.shape)


def _entropy(p):
    p = p.reshape(-1)
    safe = jnp.clip(p, 1e-300, 1.0)
    return -jnp.sum(jnp.where(p > 0, p * jnp.log(safe), 0.0))


def objective_j(pi, q, alpha):
    """J(pi) = KL(pi || q) + alpha * sum_i H(pi_i), eq 18."""
    pi = jnp.asarray(pi)
    q = jnp.asarray(q)
    outside = jnp.any((pi > 1e-15) & (q <= 0))
    log_pi = jnp.log(jnp.clip(pi, 1e-300, 1.0))
    log_q = jnp.log(jnp.clip(q, 1e-300, 1.0))
    kl = jnp.sum(jnp.where(pi > 0, pi * (log_pi - log_q), 0.0))
    entropy_sum = _entropy(pi.sum(axis=1)) + _entropy(pi.sum(axis=0))
    val = kl + alpha * entropy_sum
    return jnp.where(outside, jnp.inf, val)


def expected_sequence_loss(p1, p2, q, alpha):
    """Expected sequence-level alpha loss of a factorized predictor.

    For alpha > 0 this is (1 - F) / alpha with F = sum_y q(y) p(y)^alpha
    and p(y) = p1(y1) p2(y2). At alpha = 0 it is the cross-entropy limit
    E[-log p(y)], infinite if any positive-q cell has predicted probability 0.
    """
    p1 = jnp.asarray(p1)
    p2 = jnp.asarray(p2)
    q = jnp.asarray(q)
    joint = p1[:, None] * p2[None, :]
    alpha = float(alpha)
    if alpha == 0.0:
        bad = jnp.any((q > 0) & (joint <= 0))
        log_joint = jnp.log(jnp.clip(joint, 1e-300, 1.0))
        nll = jnp.sum(jnp.where(q > 0, -q * log_joint, 0.0))
        return jnp.where(bad, jnp.inf, nll)
    log_joint = jnp.log(jnp.clip(joint, 1e-300, 1.0))
    contrib = jnp.where((q > 0) & (joint > 0), q * jnp.exp(alpha * log_joint), 0.0)
    f_alpha = jnp.sum(contrib)
    return (1.0 - f_alpha) / alpha


def expected_tokenwise_loss(p1, p2, q, alpha):
    """Expected sum of per-token alpha losses. At alpha = 1 this fits token modes."""
    p1 = jnp.asarray(p1)
    p2 = jnp.asarray(p2)
    q = jnp.asarray(q)
    m1, m2 = marginals(q)
    return _token_population(p1, m1, alpha) + _token_population(p2, m2, alpha)


def _token_population(p, marg, alpha):
    alpha = float(alpha)
    if alpha == 0.0:
        bad = jnp.any((marg > 0) & (p <= 0))
        nll = -jnp.sum(
            jnp.where(marg > 0, marg * jnp.log(jnp.clip(p, 1e-300, 1.0)), 0.0)
        )
        return jnp.where(bad, jnp.inf, nll)
    powered = jnp.where(p > 0, jnp.exp(alpha * jnp.log(jnp.clip(p, 1e-300, 1.0))), 0.0)
    return jnp.sum(marg * (1.0 - powered) / alpha)


def valid_groups(q, tol=0.0):
    """Cartesian groups of the positive support, as bipartite connected components.

    Returns a list of (row_indices, col_indices). On Figure 2 this is
    {(I,) x (am,)} and {(he, she) x (is,)}, discovered from q, not from names.
    """
    q = jnp.asarray(q)
    v1, v2 = int(q.shape[0]), int(q.shape[1])
    positive = [
        (i, j) for i in range(v1) for j in range(v2) if float(q[i, j]) > tol
    ]
    parent = list(range(v1 + v2))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in positive:
        union(i, v1 + j)
    comps = {}
    for i, j in positive:
        root = find(i)
        comps.setdefault(root, (set(), set()))
        comps[root][0].add(i)
        comps[root][1].add(j)
    groups = [
        (tuple(sorted(rows)), tuple(sorted(cols))) for rows, cols in comps.values()
    ]
    groups.sort()
    return groups


def _simplex_grid(n, resolution, dtype):
    counts = []

    def rec(prefix, left, bins_left):
        if bins_left == 1:
            counts.append(prefix + [left])
            return
        for i in range(left + 1):
            rec(prefix + [i], left - i, bins_left - 1)

    rec([], int(resolution), int(n))
    return jnp.asarray(counts, dtype=dtype) / float(resolution)


def _best(candidates):
    best = None
    best_loss = jnp.inf
    for item in candidates:
        loss = item[-1]
        if float(loss) < float(best_loss):
            best = item
            best_loss = loss
    return best


def _adam_scan(loss_fn, params, steps, lr):
    m0 = tree_map(jnp.zeros_like, params)
    v0 = tree_map(jnp.zeros_like, params)

    def step(state, _unused):
        theta, m, v, t = state
        _loss, grads = jax.value_and_grad(loss_fn)(theta)
        grads = tree_map(lambda g: jnp.clip(g, -20.0, 20.0), grads)
        t = t + 1
        m = tree_map(lambda mi, g: 0.9 * mi + 0.1 * g, m, grads)
        v = tree_map(lambda vi, g: 0.999 * vi + 0.001 * (g * g), v, grads)
        mhat = tree_map(lambda mi: mi / (1.0 - 0.9**t), m)
        vhat = tree_map(lambda vi: vi / (1.0 - 0.999**t), v)
        frac = 1.0 - t / float(steps)
        lr_t = lr * (0.05 + 0.95 * frac)
        theta = tree_map(
            lambda p, mh, vh: p - lr_t * mh / (jnp.sqrt(vh) + 1e-8),
            theta,
            mhat,
            vhat,
        )
        return (theta, m, v, t), None

    state, _ = jax.lax.scan(
        step,
        (params, m0, v0, jnp.array(0, dtype=jnp.int32)),
        None,
        length=int(steps),
    )
    return state[0]


def _logits_from_probs(p, eps=1e-5):
    p = jnp.clip(p, eps, 1.0)
    p = p / jnp.sum(p)
    return jnp.log(p)


def _vertex_candidates(q, loss_of):
    dtype = q.dtype
    out = []
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            p1 = jnp.zeros(q.shape[0], dtype=dtype).at[i].set(1.0)
            p2 = jnp.zeros(q.shape[1], dtype=dtype).at[j].set(1.0)
            out.append((p1, p2, loss_of(p1, p2)))
    return out


def _grid_best(q, loss_of, resolution):
    g1 = _simplex_grid(q.shape[0], resolution, q.dtype)
    g2 = _simplex_grid(q.shape[1], resolution, q.dtype)

    def row_losses(p1):
        return jax.vmap(lambda p2: loss_of(p1, p2))(g2)

    losses = jax.vmap(row_losses)(g1)
    losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
    flat = int(jnp.argmin(losses))
    i = flat // int(g2.shape[0])
    j = flat % int(g2.shape[0])
    return g1[i], g2[j], losses[i, j]


def _embed_group(local1, local2, rows, cols, shape, dtype):
    p1 = jnp.zeros(shape[0], dtype=dtype).at[jnp.asarray(rows)].set(local1)
    p2 = jnp.zeros(shape[1], dtype=dtype).at[jnp.asarray(cols)].set(local2)
    return p1, p2


def _grid_group(q, loss_of, rows, cols, resolution):
    g1 = _simplex_grid(len(rows), resolution, q.dtype)
    g2 = _simplex_grid(len(cols), resolution, q.dtype)

    def row_losses(local1):
        def one(local2):
            p1, p2 = _embed_group(local1, local2, rows, cols, q.shape, q.dtype)
            return loss_of(p1, p2)

        return jax.vmap(one)(g2)

    losses = jax.vmap(row_losses)(g1)
    losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
    flat = int(jnp.argmin(losses))
    i = flat // int(g2.shape[0])
    j = flat % int(g2.shape[0])
    p1, p2 = _embed_group(g1[i], g2[j], rows, cols, q.shape, q.dtype)
    return p1, p2, losses[i, j]


def _polish_full(q, loss_of, init_p1, init_p2, steps, lr):
    init = (_logits_from_probs(init_p1), _logits_from_probs(init_p2))

    def loss_fn(params):
        return loss_of(jax.nn.softmax(params[0]), jax.nn.softmax(params[1]))

    logits = _adam_scan(loss_fn, init, steps, lr)
    p1 = jax.nn.softmax(logits[0])
    p2 = jax.nn.softmax(logits[1])
    return p1, p2, loss_of(p1, p2)


def _polish_group(q, loss_of, rows, cols, init_p1, init_p2, steps, lr):
    rows_a = jnp.asarray(rows, dtype=jnp.int32)
    cols_a = jnp.asarray(cols, dtype=jnp.int32)
    init = (
        _logits_from_probs(init_p1[rows_a]),
        _logits_from_probs(init_p2[cols_a]),
    )

    def loss_fn(params):
        p1, p2 = _embed_group(
            jax.nn.softmax(params[0]),
            jax.nn.softmax(params[1]),
            rows_a,
            cols_a,
            q.shape,
            q.dtype,
        )
        return loss_of(p1, p2)

    logits = _adam_scan(loss_fn, init, steps, lr)
    p1, p2 = _embed_group(
        jax.nn.softmax(logits[0]),
        jax.nn.softmax(logits[1]),
        rows_a,
        cols_a,
        q.shape,
        q.dtype,
    )
    return p1, p2, loss_of(p1, p2)


def _snap_pair(p1, p2, loss_of, tol=1e-8):
    s1 = jnp.where(p1 < tol, 0.0, p1)
    s2 = jnp.where(p2 < tol, 0.0, p2)
    if float(jnp.sum(s1)) <= 0.0 or float(jnp.sum(s2)) <= 0.0:
        return p1, p2
    s1 = s1 / jnp.sum(s1)
    s2 = s2 / jnp.sum(s2)
    if float(loss_of(s1, s2)) <= float(loss_of(p1, p2)) + 1e-12:
        return s1, s2
    return p1, p2


def _minimize_factors(q, loss_of, resolution, group_resolution, steps, lr):
    q = jnp.asarray(q, dtype=jnp.float64)
    candidates = []
    candidates.extend(_vertex_candidates(q, loss_of))
    candidates.append(_grid_best(q, loss_of, resolution))
    for rows, cols in valid_groups(q):
        candidates.append(_grid_group(q, loss_of, rows, cols, group_resolution))
    if steps > 0:
        uniform1 = jnp.ones(q.shape[0], dtype=q.dtype) / q.shape[0]
        uniform2 = jnp.ones(q.shape[1], dtype=q.dtype) / q.shape[1]
        candidates.append(_polish_full(q, loss_of, uniform1, uniform2, steps, lr))
        best_p1, best_p2, _best_loss = _best(candidates)
        candidates.append(_polish_full(q, loss_of, best_p1, best_p2, steps, lr))
        for rows, cols in valid_groups(q):
            candidates.append(
                _polish_group(q, loss_of, rows, cols, best_p1, best_p2, steps, lr)
            )
    p1, p2, loss = _best(candidates)
    p1, p2 = _snap_pair(p1, p2, loss_of)
    p1 = jnp.clip(p1, 0.0, None)
    p2 = jnp.clip(p2, 0.0, None)
    p1 = p1 / jnp.sum(p1)
    p2 = p2 / jnp.sum(p2)
    return p1, p2, loss


def minimize_factorized_sequence(
    q,
    alpha,
    resolution=_GRID_RESOLUTION,
    group_resolution=_GROUP_RESOLUTION,
    steps=_POLISH_STEPS,
    lr=_POLISH_LR,
):
    """Minimize expected sequence-level alpha loss. Returns (p1, p2, loss)."""
    q = jnp.asarray(q, dtype=jnp.float64)
    alpha = float(alpha)

    def loss_of(p1, p2):
        return expected_sequence_loss(p1, p2, q, alpha)

    return _minimize_factors(q, loss_of, resolution, group_resolution, steps, lr)


def minimize_factorized_tokenwise(
    q,
    alpha,
    resolution=_GRID_RESOLUTION,
    group_resolution=_GROUP_RESOLUTION,
    steps=_POLISH_STEPS,
    lr=_POLISH_LR,
):
    """Minimize expected token-wise alpha loss. Returns (p1, p2, loss)."""
    q = jnp.asarray(q, dtype=jnp.float64)
    alpha = float(alpha)

    def loss_of(p1, p2):
        return expected_tokenwise_loss(p1, p2, q, alpha)

    return _minimize_factors(q, loss_of, resolution, group_resolution, steps, lr)


def _support_index(q):
    return jnp.flatnonzero(q.reshape(-1) > 0)


def _pi_from_probs(probs, flat_index, shape, dtype):
    pi = jnp.zeros(int(shape[0] * shape[1]), dtype=dtype).at[flat_index].set(probs)
    return pi.reshape(shape)


def _grid_best_pi(q, alpha, flat_index, resolution):
    grid = _simplex_grid(int(flat_index.shape[0]), resolution, q.dtype)

    def loss_of_row(probs):
        pi = _pi_from_probs(probs, flat_index, q.shape, q.dtype)
        return objective_j(pi, q, alpha)

    losses = jax.vmap(loss_of_row)(grid)
    losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
    idx = int(jnp.argmin(losses))
    pi = _pi_from_probs(grid[idx], flat_index, q.shape, q.dtype)
    return pi, losses[idx]


def _polish_pi(q, alpha, flat_index, init_probs, steps, lr):
    init = _logits_from_probs(init_probs)

    def loss_fn(logits):
        pi = _pi_from_probs(jax.nn.softmax(logits), flat_index, q.shape, q.dtype)
        return objective_j(pi, q, alpha)

    logits = _adam_scan(loss_fn, init, steps, lr)
    pi = _pi_from_probs(jax.nn.softmax(logits), flat_index, q.shape, q.dtype)
    return pi, objective_j(pi, q, alpha)


def _face_index(q, rows, cols, flat_index):
    coord = [(i, j) for i in range(q.shape[0]) for j in range(q.shape[1])]
    row_set, col_set = set(rows), set(cols)
    face = []
    for flat_i in flat_index:
        i, j = coord[int(flat_i)]
        if i in row_set and j in col_set:
            face.append(int(flat_i))
    return jnp.asarray(face, dtype=jnp.int32)


def minimize_j(
    q,
    alpha,
    resolution=40,
    group_resolution=_GROUP_RESOLUTION,
    steps=_POLISH_STEPS,
    lr=_POLISH_LR,
):
    """Minimize J over joints supported on q. Returns (pi, loss)."""
    q = jnp.asarray(q, dtype=jnp.float64)
    alpha = float(alpha)
    flat_index = _support_index(q)
    candidates = []
    n_support = int(flat_index.shape[0])
    for n in range(n_support):
        probs = jnp.zeros(n_support, dtype=q.dtype).at[n].set(1.0)
        pi = _pi_from_probs(probs, flat_index, q.shape, q.dtype)
        candidates.append((pi, objective_j(pi, q, alpha)))
    candidates.append(_grid_best_pi(q, alpha, flat_index, resolution))
    for rows, cols in valid_groups(q):
        face = _face_index(q, rows, cols, flat_index)
        if int(face.shape[0]) == 0:
            continue
        candidates.append(_grid_best_pi(q, alpha, face, group_resolution))
    if steps > 0:
        best_pi, _loss = _best(candidates)
        support_probs = best_pi.reshape(-1)[flat_index]
        # If the current best sits on a face, also polish that face.
        candidates.append(_polish_pi(q, alpha, flat_index, support_probs, steps, lr))
        for rows, cols in valid_groups(q):
            face = _face_index(q, rows, cols, flat_index)
            if int(face.shape[0]) < 2:
                continue
            face_probs = best_pi.reshape(-1)[face]
            if float(jnp.sum(face_probs)) <= 0.0:
                face_probs = jnp.ones(face.shape[0], dtype=q.dtype) / face.shape[0]
            else:
                face_probs = face_probs / jnp.sum(face_probs)
            candidates.append(_polish_pi(q, alpha, face, face_probs, steps, lr))
    pi, loss = _best(candidates)
    snapped = jnp.where(pi < 1e-8, 0.0, pi)
    if float(jnp.sum(snapped)) > 0.0:
        snapped = snapped / jnp.sum(snapped)
        snapped_loss = objective_j(snapped, q, alpha)
        if float(snapped_loss) <= float(loss) + 1e-12:
            pi, loss = snapped, snapped_loss
    return pi, loss
