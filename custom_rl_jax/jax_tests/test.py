import jax
from jax import numpy as jnp, Array, random
from jax.typing import ArrayLike
from ..networks.mlp import Mlp

key = random.PRNGKey(234)

key, init_key, action_key = random.split(key, 3)

network_input = jnp.ones((4,), dtype=jnp.float32)

actor_model = Mlp(features=[64, 64, 4], last_layer_scale=0.01)
params = actor_model.init(init_key, network_input)


def act(actor_params, obs: ArrayLike, key: random.KeyArray) -> Array:
    logits = actor_model.apply(actor_params, obs)
    return random.categorical(key, logits)


def vectorized_act(actor_params, obs: ArrayLike, key: random.KeyArray) -> tuple[Array, random.KeyArray]:
    keys = random.split(key, obs.shape[0] + 1)
    actions = jax.vmap(act, in_axes=(None, 0, 0))(actor_params, obs, keys[:-1])
    return actions, keys[-1]


# print(jax.make_jaxpr(act)(params, network_input, action_key))

lowered = jax.jit(act).lower(params, network_input, action_key)
#print(lowered.as_text())
compiled = lowered.compile()
#
print(compiled.as_text())
