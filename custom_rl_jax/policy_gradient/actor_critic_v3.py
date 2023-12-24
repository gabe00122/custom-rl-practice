import jax
from jax.typing import ArrayLike
from jax import numpy as jnp, random, Array
import flax.linen as nn
from typing_extensions import TypedDict

from flax.training.train_state import TrainState

Params = TypedDict("Params", {
    'discount': float,
    'actor_training_state': TrainState,
    'critic_training_state': TrainState,
})

Metrics = TypedDict("Metrics", {
    'state_value': Array,
    'td_error': Array,
    'actor_loss': Array,
    'critic_loss': Array,
})


def l2_regularization(params, alpha):
    return alpha * sum(alpha * (p ** 2).mean() for p in jax.tree_leaves(params))


def actor_critic_v3(actor_model: nn.Module, critic_model: nn.Module):
    def vectorized_act(actor_params, obs: ArrayLike, key: random.KeyArray) -> tuple[Array, random.KeyArray]:
        keys = random.split(key, obs.shape[0] + 1)
        actions = jax.vmap(act, in_axes=(None, 0, 0))(actor_params, obs, keys[:-1])
        return actions, keys[-1]

    def act(actor_params, obs: ArrayLike, key: random.KeyArray) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return random.categorical(key, logits)

    def action_log_prob(actor_params, obs: ArrayLike, action: ArrayLike) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return jnp.log(nn.softmax(logits)[action])

    def state_value(critic_params, obs: ArrayLike) -> Array:
        return critic_model.apply(critic_params, obs)

    def critic_loss(critic_params, obs: ArrayLike, expected_values: ArrayLike) -> Array:
        critic_values = jax.vmap(state_value, (None, 0))(critic_params, obs).flatten()
        # jax.debug.print("critic_values: {}", critic_values)
        # jax.debug.print("expected_values: {}", expected_values)
        # jax.debug.print("critic_values - expected_values: {}", critic_values - expected_values)
        # jax.debug.print("(critic_values - expected_values) ** 2: {}", (critic_values - expected_values) ** 2)

        loss = ((expected_values - critic_values) ** 2).mean()
        # jax.debug.print("loss: {}", loss)
        # loss += l2_regularization(critic_params, alpha=0.001)
        return loss

    def update_critic(critic_training_state: TrainState, obs: ArrayLike, expected_values: ArrayLike):
        critic_params = critic_training_state.params
        loss, gradient = jax.value_and_grad(critic_loss)(critic_params, obs, expected_values)
        return critic_training_state.apply_gradients(grads=gradient), loss

    def actor_loss(actor_params, states, actions, advantages):
        action_probs = jax.vmap(action_log_prob, (None, 0, 0))(actor_params, states, actions)
        #jax.debug.print("action_probs: {}", action_probs)

        loss = -jnp.mean(action_probs * advantages)
        loss += l2_regularization(actor_params, alpha=0.0001)
        return loss

    def update_actor(actor_training_state: TrainState, states, actions, advantages):
        loss, gradient = jax.value_and_grad(actor_loss)(actor_training_state.params, states, actions, advantages)
        return actor_training_state.apply_gradients(grads=gradient), loss

    def vectorized_train_step(params: Params, obs: ArrayLike, action: ArrayLike, next_obs: ArrayLike, rewards: ArrayLike,
                              done: ArrayLike, importance: ArrayLike, key: random.KeyArray) -> tuple[
                              Params, Array, Array, random.KeyArray, Metrics]:
        discount = params['discount']
        critic_training_state = params['critic_training_state']
        critic_params = critic_training_state.params
        actor_training_state = params['actor_training_state']

        v_state_value = jax.vmap(state_value, (None, 0))

        expected_values = v_state_value(critic_params, obs).flatten()
        #jax.debug.print("expected_values: {}", expected_values)

        next_expected_values = jnp.where(done,
                                         rewards,
                                         rewards + discount * v_state_value(critic_params, next_obs).flatten())
        #jax.debug.print("next_expected_values: {}", next_expected_values)

        advantages = next_expected_values - expected_values
        advantages *= importance
        #jax.debug.print("advantages: {}", advantages)
        #jax.debug.print("importance: {}", importance)

        critic_training_state, c_loss = update_critic(critic_training_state, obs, next_expected_values)
        actor_training_state, a_loss = update_actor(actor_training_state, obs, action, advantages)

        # set the importance back to 1 if it's the end of an episode
        importance *= discount
        importance = jnp.maximum(importance, done)
        output_params: Params = {
            'discount': discount,
            'critic_training_state': critic_training_state,
            'actor_training_state': actor_training_state,
        }
        actions, key = vectorized_act(actor_training_state.params, next_obs, key)

        metrics: Metrics = {
            'actor_loss': a_loss,
            'critic_loss': c_loss,
            'td_error': jnp.mean(advantages),
            'state_value': jnp.mean(expected_values),
        }

        return output_params, importance, actions, key, metrics

    # return vectorized_train_step, vectorized_act, act
    return jax.jit(vectorized_train_step), jax.jit(vectorized_act), jax.jit(act)
