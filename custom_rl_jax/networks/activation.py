import flax.linen as nn
import jax.numpy as jnp


def mish(x: jnp.ndarray):
    return x * jnp.tanh(nn.softplus(x))
