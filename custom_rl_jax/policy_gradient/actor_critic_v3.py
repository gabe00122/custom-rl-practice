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


def actor_critic_v3(actor_model: nn.Module, critic_model: nn.Module, vectorized_count: int):
    # @jax.jit
    def vectorized_act(actor_params, obs: ArrayLike, key: random.KeyArray) -> tuple[Array, random.KeyArray]:
        keys = random.split(key, obs.shape[0] + 1)
        actions = jax.vmap(act, in_axes=(None, 0, 0))(actor_params, obs, keys[:-1])
        return actions, keys[-1]

    # @jax.jit
    def act(actor_params, obs: ArrayLike, key: random.KeyArray) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return random.categorical(key, logits)

    # @jax.jit
    def action_prob(actor_params, obs: ArrayLike, action: ArrayLike) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return nn.log_softmax(logits)[action]

    # @jax.jit
    def state_value(critic_params, obs: ArrayLike) -> Array:
        return critic_model.apply(critic_params, obs).reshape(())

    # @jax.jit
    def critic_loss(critic_params, obs: ArrayLike, expected_values: ArrayLike) -> Array:
        critic_values = jax.vmap(critic_model.apply, (None, 0))(critic_params, obs)
        loss = ((expected_values - critic_values) ** 2).mean()
        return loss

    # @jax.jit
    def update_critic(critic_training_state: TrainState, obs: ArrayLike, expected_values: ArrayLike):
        critic_params = critic_training_state.params
        loss, gradient = jax.value_and_grad(critic_loss)(critic_params, obs, expected_values)
        return critic_training_state.apply_gradients(grads=gradient), loss

    def l2_loss(x, alpha):
        return alpha * (x ** 2).mean()

    # @jax.jit
    def actor_loss(actor_params, states, actions, advantages):
        action_probs = jax.vmap(action_prob, (None, 0, 0))(actor_params, states, actions)

        loss = -jnp.mean(action_probs * advantages)
        loss += sum(
            l2_loss(w, alpha=0.001)
            for w in jax.tree_leaves(actor_params)
        )
        return loss

    # @jax.jit
    def update_actor(actor_training_state: TrainState, states, actions, advantages):
        loss, gradient = jax.value_and_grad(actor_loss)(actor_training_state.params, states, actions, advantages)
        return actor_training_state.apply_gradients(grads=gradient), loss

    # @jax.jit
    def vectorized_train_step(params: Params, obs: ArrayLike, action: ArrayLike, next_obs: ArrayLike, reward: ArrayLike,
                              done: ArrayLike, importance: ArrayLike, key: random.KeyArray) -> tuple[Params, Array, Metrics]:
        discount = params['discount']
        critic_training_state = params['critic_training_state']
        critic_params = critic_training_state.params
        actor_training_state = params['actor_training_state']
        actor_params = actor_training_state.params

        v_state_value = jax.vmap(state_value, (None, 0))

        expected_values = v_state_value(critic_params, obs)
        next_expected_values = jnp.where(done,
                                         reward,
                                         reward + discount * v_state_value(critic_params, next_obs))

        advantages = next_expected_values - expected_values
        advantages *= importance

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
        actions, key = vectorized_act(actor_params, next_obs, key)

        metrics: Metrics = {
            'actor_loss': a_loss,
            'critic_loss': c_loss,
            'td_error': jnp.mean(advantages),
            'state_value': jnp.mean(expected_values),
        }

        return output_params, importance, actions, key, metrics

    # return vectorized_train_step, vectorized_act, act
    return jax.jit(vectorized_train_step), jax.jit(vectorized_act), jax.jit(act)
