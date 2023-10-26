from typing import Any, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import utils

from .utils import Decomposition


def diag_pinv(a):
    max_rows_cols = a.shape[-1]
    rcond = 10.0 * max_rows_cols * jnp.array(jnp.finfo(a.dtype).eps)
    cutoff = rcond * jnp.abs(a).max()
    a = jnp.where(a > cutoff, a, jnp.inf).astype(a.dtype)
    a = jnp.reciprocal(a)

    return a


def bias_correction(decay, count, vector):
    return jax.tree_map(lambda e: (e / (1 - decay**count)).astype(e.dtype), vector)


def ginger(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    tau: int = 4,
    damping: float = 1e-8,
    rho: Optional[float] = None,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    return scale_by_ginger(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        tau=tau,
        damping=damping,
        rho=rho,
        mu_dtype=mu_dtype,
    )


class ScaleByGingerState(NamedTuple):
    """State for the Lion algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates  # pytree of arrays, dtype=jnp.floating.

    decomposition: chex.ArrayTree


def scale_by_ginger(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    tau: int = 4,
    damping: float = 1e-8,
    rho: Optional[float] = None,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        def fill_diag(nrow, ncol):
            nrow = jnp.maximum(nrow, 1)
            idr, idc = jnp.diag_indices(nrow)
            idc = jnp.mod(idc, ncol)
            mat = jnp.zeros((nrow, ncol))
            mat = mat.at[idr, idc].set(1.0)
            mat = mat / jnp.sqrt(jnp.sum(mat**2, axis=0))
            return mat

        flattened, _ = jax.flatten_util.ravel_pytree(params)
        return ScaleByGingerState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree_map(lambda p: jnp.zeros_like(p, dtype=mu_dtype), params),
            decomposition=jax.tree_map(
                lambda p: Decomposition(
                    matu=jnp.eye(p.size, tau),
                    sigma=jnp.zeros((tau,), dtype=jnp.float32),
                ),
                params,
            ),
        )

    def svd_update(matu, sigma, u, a, b):
        precision = jnp.float32

        p = u - matu @ (matu.T @ u)  # shape=(d,)
        r = jnp.sqrt(p.T @ p)
        r = jnp.where(r < 1e-8, 1e-8, r)  # Avoid division by zero.
        p = jnp.true_divide(p, r)
        matk_l = jnp.zeros((tau + 1, tau + 1), dtype=precision)
        matk_l = jax.lax.dynamic_update_slice(matk_l, jnp.eye(tau, dtype=precision), (0, 0))
        matk_l = jax.lax.dynamic_update_slice(matk_l, (matu.T @ u).reshape(-1, 1).astype(precision), (0, tau))
        matk_l = jax.lax.dynamic_update_slice(matk_l, jnp.array([[r]], dtype=precision), (tau, tau))
        matk_m = jnp.zeros((tau + 1, tau + 1), dtype=precision)
        matk_m = jax.lax.dynamic_update_slice(matk_m, jnp.diag(sigma * a).astype(precision), (0, 0))
        matk_m = jax.lax.dynamic_update_slice(matk_m, jnp.array([[b]], dtype=precision), (tau, tau))
        matv, sigma, _ = jnp.linalg.svd((matk_l @ matk_m @ matk_l.T).astype(precision))

        # lower the precisions back
        matv = matv.astype(mu_dtype)
        matu = matu @ matv[:tau] + jnp.outer(p, matv[tau])
        return Decomposition(matu=matu, sigma=sigma)

    def query(state, tree_array, scale=1.0, power=1.0):
        def block_query(decomposition, u):
            u, pytree = jax.flatten_util.ravel_pytree(u)
            sb = scale * jnp.power(decomposition.sigma, power)
            sprime = diag_pinv(damping**2 + damping * sb) * sb

            return pytree(u / damping - decomposition.matu @ (jnp.diag(sprime) @ (decomposition.matu.T @ u)))

        return jax.tree_map(
            block_query, state.decomposition, tree_array, is_leaf=lambda x: isinstance(x, Decomposition)
        )

    def update_state(state, grads, ngrads):
        a = b2
        b = 1 - b2

        def block_update(decomposition, q, ng):
            q, _ = jax.flatten_util.ravel_pytree(q)
            ng, _ = jax.flatten_util.ravel_pytree(ng)

            gamma = b / (1 + b * q @ ng)

            sb = a * decomposition.sigma
            sprime = diag_pinv(damping**2 + damping * sb) * sb

            new_decomp = svd_update(decomposition.matu, sprime, q, 1.0, gamma)
            matu = new_decomp.matu
            sigma = new_decomp.sigma

            sigma = jnp.where(sigma > 1 / damping, 1 / damping, sigma)

            sigma = (damping**2) * sigma * diag_pinv(1 - damping * sigma)

            sigma = jnp.where(sigma < 1e-9 * sigma.max(), 0.0, sigma)
            sorted_idx = jnp.argsort(sigma)
            sigma = sigma[sorted_idx]
            matu = matu[:, sorted_idx]
            matu = jax.lax.dynamic_slice(matu, (0, 1), (matu.shape[0], tau))
            sigma = jax.lax.dynamic_slice(sigma, (1,), (tau,))

            return Decomposition(matu=matu, sigma=sigma)

        decomposition = jax.tree_map(
            block_update,
            state.decomposition,
            query(state, ngrads, scale=a),
            ngrads,
            is_leaf=lambda x: isinstance(x, Decomposition),
        )

        return ScaleByGingerState(
            count=optax.safe_int32_increment(state.count),
            mu=jax.tree_map(lambda x, y: (b1 * x + (1 - b1) * y).astype(mu_dtype), state.mu, grads),
            decomposition=decomposition,
        )

    def update_fn(grads, state, ngrads):
        new_state = update_state(state, grads, ngrads)
        updates = query(
            new_state,
            tree_array=bias_correction(b1, new_state.count, new_state.mu),
            scale=1.0,
            power=1.0,
        )

        if rho is not None:
            updates = jax.tree_map(lambda x: jnp.clip(x, -rho, rho), updates)

        lr = learning_rate(new_state.count) if callable(learning_rate) else learning_rate
        updates = jax.tree_map(lambda x: x * -lr, updates)

        # return the updates and the new state
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
