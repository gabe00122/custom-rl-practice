import jax
from jax import numpy as jnp, random
import flax.linen as nn
import gymnasium as gym
from typing import Any
from .train import Metrics, HyperParams


def actor_critic(actor_model: nn.Module, critic_model: nn.Module):
    @jax.jit
    def act(actor_params, state: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        return random.categorical(key, logits)

    @jax.jit
    def action_prob(actor_params, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        return nn.softmax(logits)[action]

    @jax.jit
    def ln_action_prob(actor_params, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(action_prob(actor_params, state, action))

    grad_ln_action_prob = jax.grad(ln_action_prob)

    @jax.jit
    def state_value(critic_params, state: jnp.ndarray) -> jnp.ndarray:
        return critic_model.apply(critic_params, state).reshape(())

    grad_state_value = jax.grad(state_value)

    @jax.jit
    def update_critic(critic_params, state, critic_learning_rate, td_error):
        step_size = critic_learning_rate * td_error

        critic_gradient = grad_state_value(critic_params, state)
        return jax.tree_map(lambda weight, grad: weight + step_size * grad,
                            critic_params,
                            critic_gradient)

    @jax.jit
    def update_actor(actor_params, state, action, actor_learning_rate, i, td_error):
        step_size = actor_learning_rate * i * td_error

        policy_gradient = grad_ln_action_prob(actor_params, state, action)
        return jax.tree_map(lambda weight, grad: weight + step_size * grad,
                            actor_params,
                            policy_gradient)

    def train_episode(env: gym.Env, actor_params: dict[str, Any], critic_params: dict[str, Any],
                      hyper_params: HyperParams, key: random.PRNGKey):
        discount = hyper_params['discount']
        actor_learning_rate = hyper_params['actor_learning_rate']
        critic_learning_rate = hyper_params['critic_learning_rate']

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

            action = act(actor_params, obs, action_key)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            # reward = reward / 500
            done = terminated or truncated

            s_value = state_value(critic_params, obs)
            td_error = reward - s_value
            if not done:
                td_error += discount * state_value(critic_params, next_obs)

            critic_params = update_critic(critic_params, obs, critic_learning_rate, td_error)
            actor_params = update_actor(actor_params, obs, action, actor_learning_rate, i, td_error)

            i *= discount
            obs = next_obs

            if not has_state:
                metrics['state_value'] = s_value
                has_state = True
            metrics['td_error'] += td_error
            metrics['reward'] += reward
            metrics['length'] += 1

        return actor_params, critic_params, key, metrics

    return train_episode, act
