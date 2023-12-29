import gymnasium as gym
import jax
import jax.numpy as jnp
import orbax.checkpoint
from pathlib import Path
from ..networks.mlp import Mlp
import flax.linen as nn
import matplotlib.pyplot as plt


def main():
    env = gym.make("LunarLander-v2")
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    print(state_space)

    actor_model = Mlp(features=[64, 64, 1], last_layer_scale=0.01)
    # critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)

    checkpoint_path = Path("./old/lander-runs/run-lander-997-5/params").absolute()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = checkpointer.restore(checkpoint_path)
    actor_params = raw_restored['critic_training_state']['params']

    def sample_probs(params, state):
        return jnp.argmax(nn.softmax(actor_model.apply(params, state)))

    def state_value(params, state):
        return actor_model.apply(params, state).reshape(())

    vectorized_sample_props = jax.vmap(sample_probs, in_axes=(None, 0))
    vectorized_state_value = jax.vmap(state_value, in_axes=(None, 0))

    width = 100
    samples = width ** 2
    zeros = jnp.zeros((samples,), dtype=jnp.float32)
    velocity = jnp.full((samples,), 0, dtype=jnp.float32)

    X, Y = jnp.mgrid[-15:15:100j, 0:20:100j]
    observations = jnp.vstack([X.ravel(), Y.ravel(), velocity, zeros, zeros, zeros, zeros, zeros]).T

    probability = vectorized_state_value(actor_params, observations)

    xy = probability.reshape((width, width))
    # rotate by -90 degrees
    xy = jnp.rot90(xy, k=1)

    plt.imshow(xy)
    plt.show()


if __name__ == '__main__':
    main()
