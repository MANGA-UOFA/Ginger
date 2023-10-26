from typing import Any, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import utils


def qng(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    tau: int = 4,
    mu_dtype: Optional[Any] = None,
    trust_region: Optional[float] = None,
) -> optax.GradientTransformation:
    return scale_by_qng(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        mu_dtype=mu_dtype,
        tau=tau,
        trust_region=trust_region,
    )


class ScaleByQNGState(NamedTuple):
    """State for the Lion algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    qs: chex.Array  # shape=(tau,d), dtype=jnp.floating.
    dots: chex.Array  # shape=(tau,), dtype=jnp.floating.
    length: chex.Array  # shape=(), dtype=jnp.int32.


def scale_by_qng(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    tau: int = 4,
    mu_dtype: Optional[Any] = None,
    trust_region: Optional[float] = None,
) -> optax.GradientTransformation:
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        qs = jax.tree_map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)

        qs = jax.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), tau, 0),
            qs,
        )

        return ScaleByQNGState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            qs=qs,
            length=jnp.zeros([], jnp.int32),
            dots=jnp.zeros((tau,), dtype=jnp.float32),
        )

    def tree_dot(a, b):
        return jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_map(lambda x, y: jnp.sum(x * y).astype(jnp.float32), a, b),
            0.0,
        )

    def _recursion(dots, qs, length, u, reverse=False):
        a = b2
        b = 1 - b2
        reverse = 1 * reverse

        def cond_fn(val):
            i, _, _ = val
            return i < tau

        def body_fn(val):
            _i, (dots, qs, u, reverse), t = val

            def true_fn(t):
                i = reverse * (length - _i - 1) + (1 - reverse) * _i
                dot = dots[i]
                q = jax.tree_map(lambda x: x[i], qs)
                scalar = (1 / dot) * (1 - jnp.sqrt(a / (a + b * dot))) * tree_dot(q, u)
                t = jax.tree_map(lambda _t, _q: (_t - scalar * _q).astype(mu_dtype), t, q)
                return t

            def false_fn(t):
                return t

            return _i + 1, (dots, qs, u, reverse), jax.lax.cond(_i < length, true_fn, false_fn, t)

        _, _, t = jax.lax.while_loop(cond_fn, body_fn, (0, (dots, qs, u, reverse), u))
        # val = (0, (dots, qs, u, reverse), u)
        # for _ in range(tau):
        #     val = body_fn(val)
        # _, _, t = val
        t = jax.tree_map(lambda x: (a ** (-length / 2) * x).astype(mu_dtype), t)
        return t

    def _update_mem(state, ng):
        new_q = _recursion(state.dots, state.qs, state.length, ng, reverse=True)
        qs = jax.tree_map(
            lambda x, y: jax.lax.dynamic_update_index_in_dim(
                jnp.roll(x, 1, axis=0),
                y[None, ...].astype(mu_dtype),
                0,
                axis=0,
            ),
            state.qs,
            new_q,
        )
        new_dot = tree_dot(new_q, new_q)
        dots = jax.tree_map(
            lambda x, y: jax.lax.dynamic_update_index_in_dim(jnp.roll(x, 1), y[None, ...], 0, axis=0),
            state.dots,
            new_dot,
        )

        length = jnp.minimum(state.length + 1, tau)
        return qs, dots, length

    def update_fn(updates, state, ngrads):
        # mu = update_moment(updates, state.mu, b1, 1)
        mu = jax.tree_map(lambda x, y: b1 * x + y, state.mu, updates)
        mu = jax.tree_map(lambda x: x.astype(mu_dtype), mu)
        count_inc = optax.safe_int32_increment(state.count)

        # updates = bias_correction(mu, b1, count_inc)
        updates = mu

        updates = _recursion(state.dots, state.qs, state.length, updates, reverse=True)

        updates = _recursion(state.dots, state.qs, state.length, updates, reverse=False)

        qs, dots, length = _update_mem(state, ngrads)

        lr = learning_rate(count_inc) if callable(learning_rate) else learning_rate

        if trust_region is not None:
            denom = tree_dot(mu, updates)
            lr = jnp.minimum(lr, jnp.sqrt(2 * trust_region / denom))

        updates = jax.tree_map(lambda x: x * -lr, updates)

        return updates, ScaleByQNGState(count=count_inc, mu=mu, qs=qs, dots=dots, length=length)

    return optax.GradientTransformation(init_fn, update_fn)
