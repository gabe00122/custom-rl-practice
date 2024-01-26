import jax.debug
from jax import numpy as jnp
from flax import linen as nn
from .activation import mish


class ActorCriticModel(nn.Module):
    root: nn.Module
    actor: nn.Module
    critic: nn.Module

    @nn.compact
    def __call__(self, inputs):
        # x = self.root(inputs)
        # x = mish(x)
        actor_logits = self.actor(inputs)
        critic_value = self.critic(inputs)

        return actor_logits, jnp.squeeze(critic_value)
