import jax
from jax import numpy as jnp, random, Array
from jax.typing import ArrayLike
import flax.linen as nn
from flax.training.train_state import TrainState
from typing_extensions import TypedDict

Params = TypedDict("Params", {
    'discount': ArrayLike,
    'actor_l2_regularization': ArrayLike,
    'actor_training_state': TrainState,
    'critic_training_state': TrainState,
    'importance': ArrayLike,
})

Metrics = TypedDict("Metrics", {
    'state_value': Array,
    'td_error': Array,
    'actor_loss': Array,
    'critic_loss': Array,
})


def l2_regularization(params, alpha: ArrayLike) -> Array:
    return alpha * sum(alpha * (p ** 2).mean() for p in jax.tree_leaves(params))


def actor_critic_v3(actor_model: nn.Module, critic_model: nn.Module, debug: bool = False):
    def act(actor_params, obs: ArrayLike, key: random.KeyArray) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return random.categorical(key, logits)

    def vectorized_act(actor_params, obs: ArrayLike, key: random.KeyArray) -> tuple[Array, random.KeyArray]:
        keys = random.split(key, obs.shape[0] + 1)
        actions = jax.vmap(act, in_axes=(None, 0, 0))(actor_params, obs, keys[:-1])
        return actions, keys[-1]

    def action_log_prob(actor_params, obs: ArrayLike, action: ArrayLike) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return jnp.log(nn.softmax(logits)[action])

    def vectorized_state_value(critic_params, obs: ArrayLike) -> Array:
        return jax.vmap(critic_model.apply, in_axes=(None, 0))(critic_params, obs).flatten()

    def critic_loss(critic_params, obs: ArrayLike, expected_values: ArrayLike) -> Array:
        critic_values = vectorized_state_value(critic_params, obs)
        loss = ((expected_values - critic_values) ** 2).mean()

        if debug:
            jax.debug.print("critic_loss - critic_values: {}", critic_values)
            jax.debug.print("critic_loss - expected_values: {}", expected_values)
            jax.debug.print("critic_loss - critic_values - expected_values: {}", critic_values - expected_values)
            jax.debug.print("critic_loss - (critic_values - expected_values) ** 2: {}",
                            (critic_values - expected_values) ** 2)
            jax.debug.print("critic_loss - loss: {}", loss)

        return loss

    def update_critic(
            critic_training_state: TrainState,
            obs: ArrayLike,
            expected_values: ArrayLike
    ) -> tuple[TrainState, Array]:
        critic_params = critic_training_state.params
        loss, gradient = jax.value_and_grad(critic_loss)(critic_params, obs, expected_values)
        return critic_training_state.apply_gradients(grads=gradient), loss

    def actor_loss(actor_params, states, actions, advantages, actor_l2_regularization: float) -> Array:
        action_probs = jax.vmap(action_log_prob, (None, 0, 0))(actor_params, states, actions)
        if debug:
            jax.debug.print("actor_loss - action_probs: {}", action_probs)

        loss = -jnp.mean(action_probs * advantages)
        loss += l2_regularization(actor_params, alpha=actor_l2_regularization)
        return loss

    def update_actor(actor_training_state: TrainState, states, actions, advantages, actor_l2_regularization) -> tuple[TrainState, Array]:
        loss, gradient = jax.value_and_grad(actor_loss)(actor_training_state.params, states, actions, advantages, actor_l2_regularization)
        return actor_training_state.apply_gradients(grads=gradient), loss

    def vectorized_train_step(
            params: Params,
            obs: ArrayLike,
            action: ArrayLike,
            next_obs: ArrayLike,
            rewards: ArrayLike,
            done: ArrayLike,
            key: random.KeyArray
    ) -> tuple[Params, Array, random.KeyArray, Metrics]:
        discount = params['discount']
        actor_l2_regularization = params['actor_l2_regularization']
        critic_training_state = params['critic_training_state']
        critic_params = critic_training_state.params
        actor_training_state = params['actor_training_state']
        importance = params['importance']

        expected_values = vectorized_state_value(critic_params, obs)

        next_expected_values = jnp.where(
            done,
            rewards,
            rewards + discount * vectorized_state_value(critic_params, next_obs)
        )

        advantages = next_expected_values - expected_values
        advantages *= importance
        if debug:
            jax.debug.print("expected_values: {}", expected_values)
            jax.debug.print("advantages: {}", advantages)
            jax.debug.print("importance: {}", importance)

        critic_training_state, c_loss = update_critic(critic_training_state, obs, next_expected_values)
        actor_training_state, a_loss = update_actor(actor_training_state, obs, action, advantages, actor_l2_regularization)

        # set the importance back to 1 if it's the end of an episode
        importance *= discount
        importance = jnp.maximum(importance, done)

        output_params: Params = {
            'discount': discount,
            'actor_l2_regularization': actor_l2_regularization,
            'critic_training_state': critic_training_state,
            'actor_training_state': actor_training_state,
            'importance': importance,
        }
        actions, key = vectorized_act(actor_training_state.params, next_obs, key)

        metrics: Metrics = {
            'actor_loss': a_loss,
            'critic_loss': c_loss,
            'td_error': jnp.mean(advantages),
            'state_value': jnp.mean(expected_values),
        }

        return output_params, actions, key, metrics

    return jax.jit(vectorized_train_step), jax.jit(vectorized_act), jax.jit(act)
