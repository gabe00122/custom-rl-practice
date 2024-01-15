import jax
from jax import numpy as jnp
from jaxtyping import Float, Int, Array, Scalar
from .simple_gridworld import Observation, Action


def encode_observation(observation: Observation) -> Float[Array, "4"]:
    return jnp.array([
        observation.goal[0],
        observation.goal[1],
        observation.position[0],
        observation.position[1]
    ], jnp.float32)


def decode_action(action: Int[Scalar, ""]) -> Action:
    return Action(action)
