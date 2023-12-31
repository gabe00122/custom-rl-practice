import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from pathlib import Path
from ..networks.mlp import Mlp
import flax.linen as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


def main():
    env = gym.make("LunarLander-v2")
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    print(state_space)

    actor_model = Mlp(features=[64, 64, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)

    checkpoint_path = Path("./old/run-lander-l2-init/params").absolute()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = checkpointer.restore(checkpoint_path)
    actor_params = raw_restored['actor_training_state']['params']
    critic_params = raw_restored['critic_training_state']['params']

    def sample_probs(params, state):
        return jnp.argmax(nn.softmax(actor_model.apply(params, state)))

    def state_value(params, state):
        return critic_model.apply(params, state).reshape(())

    vectorized_sample_props = jax.vmap(sample_probs, in_axes=(None, 0))
    vectorized_state_value = jax.vmap(state_value, in_axes=(None, 0))

    width = 100
    samples = width ** 2
    zeros = jnp.zeros((samples,), dtype=jnp.float32)
    velocity = jnp.full((samples,), 0, dtype=jnp.float32)

    # crate two subplots for the actor and critic

    fig, (ax, ax2) = plt.subplots(1, 2)

    fig.subplots_adjust(left=0.30, bottom=0.30)

    axhor = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    hor_slider = Slider(
        ax=axhor,
        label='Hor Velocity',
        valmin=-5,
        valmax=5,
    )

    axvert = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    vert_slider = Slider(
        ax=axvert,
        label='Vert Velocity',
        valmin=-5,
        valmax=5,
    )

    axangle = fig.add_axes([0.25, 0.2, 0.65, 0.03])
    angle_slider = Slider(
        ax=axangle,
        label='Angle',
        valmin=-3.1415927,
        valmax=3.1415927,
    )

    axangle_vel = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    angle_vel_slider = Slider(
        ax=axangle_vel,
        label='Angle Velocity',
        valmin=-5,
        valmax=5,
    )

    @jax.jit
    def sample(hor_val, vert, angle, angle_vel):
        X, Y = jnp.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
        observations = jnp.vstack([
            X.ravel(),
            Y.ravel(),
            jnp.full((samples,), hor_val),
            jnp.full((samples,), vert),
            jnp.full((samples,), angle),
            jnp.full((samples,), angle_vel),
            zeros,
            zeros,
        ]).T

        state_values = vectorized_state_value(critic_params, observations)
        state_values = state_values.reshape((width, width))
        state_values = jnp.rot90(state_values, k=1)

        sample_props = vectorized_sample_props(actor_params, observations)
        sample_props = sample_props.reshape((width, width))
        sample_props = jnp.rot90(sample_props, k=1)

        return state_values, sample_props

    def update(val):
        sv, sp = sample(hor_slider.val, vert_slider.val, angle_slider.val, angle_vel_slider.val)
        # rotate by -90 degrees

        ax.imshow(sv, vmin=-100, vmax=100)
        ax2.imshow(sp, vmin=0, vmax=3)
        fig.canvas.draw_idle()

    hor_slider.on_changed(update)
    vert_slider.on_changed(update)
    angle_slider.on_changed(update)
    angle_vel_slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()
