import jax
from jax import numpy as jnp, Array
from jax.typing import ArrayLike


def l2_init_regularization(params, original_params, alpha: ArrayLike) -> Array:
    delta = jax.tree_map(lambda p, op: p - op, params, original_params)
    return l2_regularization(delta, alpha)


def l2_regularization(params, alpha: ArrayLike) -> Array:
    leaves = jax.tree_util.tree_leaves(params)
    return alpha * sum(jnp.sum(jnp.square(p)) for p in leaves)


def mul_exp(x, logp):
    p = jnp.exp(logp)
    x = jnp.where(p == 0, 0.0, x)
    return x * p


def entropy_loss(action_probs) -> Array:
    return jnp.sum(mul_exp(action_probs, action_probs))
