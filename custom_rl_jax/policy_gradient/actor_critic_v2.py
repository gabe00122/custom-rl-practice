import jax
from jax import numpy as jnp, random
import flax.linen as nn
import gymnasium as gym
from typing_extensions import TypedDict
from .types import Metrics

import optax
from flax.training.train_state import TrainState

Params = TypedDict("Params", {
    'discount': float,
    'actor_training_state': TrainState,
    'critic_training_state': TrainState,
})

actor_optimizer = optax.adam(0.0001)
critic_optimizer = optax.adam(0.0001)


def actor_critic_v2(actor_model: nn.Module, critic_model: nn.Module):
    @jax.jit
    def act(actor_params, state: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        return random.categorical(key, logits)

    @jax.jit
    def action_prob(actor_params, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        return nn.softmax(logits)[action]

    @jax.jit
    def state_value(critic_params, state: jnp.ndarray) -> jnp.ndarray:
        return critic_model.apply(critic_params, state).reshape(())

    @jax.jit
    def critic_loss(critic_params, states: jnp.ndarray, expected_values: jnp.ndarray) -> jnp.ndarray:
        critic_values = critic_model.apply(critic_params, states)
        return ((expected_values - critic_values) ** 2).mean()

    grad_critic_loss = jax.grad(critic_loss)

    @jax.jit
    def update_critic(critic_training_state: TrainState, states: jnp.ndarray, expected_values: jnp.ndarray):
        critic_params = critic_training_state.params
        gradient = grad_critic_loss(critic_params, states, expected_values)
        return critic_training_state.apply_gradients(grads=gradient)

    @jax.jit
    def actor_loss(actor_params, states, actions, advantages):
        action_probs = action_prob(actor_params, states, actions)
        return -jnp.mean(jnp.log(action_probs) * advantages)

    grad_actor_loss = jax.grad(actor_loss)

    @jax.jit
    def update_actor(actor_training_state: TrainState, states, actions, advantages):
        gradient = grad_actor_loss(actor_training_state.params, states, actions, advantages)
        return actor_training_state.apply_gradients(grads=gradient)

    def train_episode(params: Params, key: random.PRNGKey, env: gym.Env) -> tuple[Metrics, Params, random.PRNGKey]:
        discount = params['discount']
        actor_training_state = params['actor_training_state']
        critic_training_state = params['critic_training_state']

        metrics: Metrics = {
            'td_error': 0,
            'state_value': 0,
            'reward': 0,
            'length': 0
        }
        has_state = False

        obs, info = env.reset()

        done = False

        i = 1.0

        while not done:
            key, action_key = random.split(key)

            action = act(actor_training_state.params, obs, action_key)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            # reward = reward / 500
            done = terminated or truncated

            s_value = state_value(critic_training_state.params, obs)
            expected_values = reward
            if not done:
                expected_values += discount * state_value(critic_training_state.params, next_obs)

            td_error = expected_values - s_value

            critic_training_state = update_critic(critic_training_state, obs, expected_values)
            actor_training_state = update_actor(actor_training_state, obs, action, td_error * i)

            i *= discount
            obs = next_obs

            if not has_state:
                metrics['state_value'] = s_value
                has_state = True
            metrics['td_error'] += td_error
            metrics['reward'] += reward
            metrics['length'] += 1

        output_params: Params = {
            'discount': discount,
            'actor_training_state': actor_training_state,
            'critic_training_state': critic_training_state,
        }

        return metrics, output_params, key

    return train_episode, act
