import gymnasium as gym
import jax
import jax.numpy as jnp
import orbax.checkpoint
from pathlib import Path
from ..networks.mlp import Mlp
from actor_critic import actor_critic
import flax.linen as nn

def main():
    env = gym.make("CartPole-v1")
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    actor_model = Mlp(features=[64, 64, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)
    _, act = actor_critic(actor_model, critic_model)

    checkpoint_path = Path("./old/cart-pole-linear/checkpoint").absolute()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = checkpointer.restore(checkpoint_path)
    actor_params = raw_restored['actor']

    @jax.jit
    def sample_props(params, state):
        return nn.softmax(actor_model.apply(params, state))

    #jnp.mgrid()


if __name__ == '__main__':
    main()
