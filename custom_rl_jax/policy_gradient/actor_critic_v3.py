import jax
from jax.typing import ArrayLike
from jax import numpy as jnp, random, Array
import flax.linen as nn
import gymnasium as gym
from typing_extensions import TypedDict
from .types import Metrics

from flax.training.train_state import TrainState

Params = TypedDict("Params", {
    'discount': float,
    'actor_training_state': TrainState,
    'critic_training_state': TrainState,
})


def mul_exp(x, logp):
    p = jnp.exp(logp)
    x = jnp.where(p == 0, 0.0, x)
    return x * p


def actor_critic_v3(actor_model: nn.Module, critic_model: nn.Module, vectorized_count: int):

    @jax.jit
    def vectorized_act(actor_params, obs: ArrayLike, key: random.KeyArray) -> tuple[Array, random.KeyArray]:
        keys = random.split(key, vectorized_count + 1)
        actions = jax.vmap(act, in_axes=(None, 0, 0), out_axes=0)(actor_params, obs, keys[:-1])
        return actions, keys[-1]

    @jax.jit
    def act(actor_params, obs: ArrayLike, key: random.KeyArray) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return random.categorical(key, logits)

    @jax.jit
    def action_prob(actor_params, obs: ArrayLike) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return nn.log_softmax(logits)

    @jax.jit
    def state_value(critic_params, obs: ArrayLike) -> Array:
        return critic_model.apply(critic_params, obs).reshape(())

    @jax.jit
    def critic_loss(critic_params, obs: ArrayLike, expected_values: ArrayLike) -> Array:
        critic_values = critic_model.apply(critic_params, obs)
        loss = ((expected_values - critic_values) ** 2).mean()
        return loss

    grad_critic_loss = jax.grad(critic_loss)

    @jax.jit
    def update_critic(critic_training_state: TrainState, states: ArrayLike, expected_values: ArrayLike):
        critic_params = critic_training_state.params
        gradient = grad_critic_loss(critic_params, states, expected_values)
        return critic_training_state.apply_gradients(grads=gradient)

    def l2_loss(x, alpha):
        return alpha * (x ** 2).mean()

    @jax.jit
    def actor_loss(actor_params, states, actions, advantages):
        action_probs = action_prob(actor_params, states)

        loss = -jnp.mean(action_probs[actions] * advantages)
        loss += sum(
            l2_loss(w, alpha=0.0001)
            for w in jax.tree_leaves(actor_params)
        )
        return loss

    grad_actor_loss = jax.grad(actor_loss)

    @jax.jit
    def update_actor(actor_training_state: TrainState, states, actions, advantages):
        gradient = grad_actor_loss(actor_training_state.params, states, actions, advantages)
        return actor_training_state.apply_gradients(grads=gradient)

    @jax.jit
    def one_step(actor_training_state, critic_training_state, obs, next_obs, action, reward, discount, i, done):

        s_value = state_value(critic_training_state.params, obs)
        expected_values = jnp.select(done, reward,
                                    reward + discount * state_value(critic_training_state.params, next_obs))

        td_error = expected_values - s_value

        critic_training_state = update_critic(critic_training_state, obs, expected_values)
        actor_training_state = update_actor(actor_training_state, obs, action, td_error * i)
        return actor_training_state, critic_training_state

    def vectorized_train_step(params: Params, obs, reward, terminated, truncated, info) -> tuple[Metrics, Params, random.PRNGKey]:
        pass

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

            actor_training_state, critic_training_state = one_step(
                actor_training_state,
                critic_training_state,
                obs,
                next_obs,
                action,
                jnp.array([reward]),
                discount,
                i,
                jnp.array([done]),
            )

            i *= discount
            obs = next_obs

            if not has_state:
                # metrics['state_value'] = s_value
                has_state = True
            # metrics['td_error'] += td_error
            metrics['reward'] += reward
            metrics['length'] += 1

        output_params: Params = {
            'discount': discount,
            'actor_training_state': actor_training_state,
            'critic_training_state': critic_training_state,
        }

        return metrics, output_params, key

    return train_episode, act, vectorized_act
