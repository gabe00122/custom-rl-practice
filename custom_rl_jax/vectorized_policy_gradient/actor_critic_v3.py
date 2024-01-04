import jax
from jax import numpy as jnp, random, Array
from jax.typing import ArrayLike
import flax.linen as nn
from flax.training.train_state import TrainState
from typing_extensions import TypedDict
from .settings import RunSettings


Params = TypedDict("Params", {
    'discount': ArrayLike,
    'actor_l2_regularization': ArrayLike,
    'entropy_regularization': ArrayLike,
    'actor_training_state': TrainState,
    'critic_training_state': TrainState,
    'importance': ArrayLike,
})

Metrics = TypedDict("Metrics", {
    'state_value': Array,
    'td_error': Array,
    'actor_loss': Array,
    'critic_loss': Array,
    'entropy_loss': Array,
    'l2_loss': Array,
})


def l2_init_regularization(params, original_params, alpha: ArrayLike) -> Array:
    delta = jax.tree_map(lambda p, op: p - op, params, original_params)
    return l2_regularization(delta, alpha)


def l2_regularization(params, alpha: ArrayLike) -> Array:
    leaves = jax.tree_util.tree_leaves(params)
    return alpha * sum(jnp.sum(jnp.square(p)) for p in leaves)


def mul_exp(x, logp):
    p = jnp.exp(logp)
    x = jnp.where(p == 0, 0.0, x)
    return x * p


def entropy_loss(action_probs) -> Array:
    return jnp.sum(mul_exp(action_probs, action_probs), axis=-1)


def actor_critic_v3(init_actor_params, init_critic_params, actor_model: nn.Module, critic_model: nn.Module, debug: bool = False):
    def act(actor_params, obs: ArrayLike, key: ArrayLike) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return random.categorical(key, logits)

    def vectorized_act(actor_params, obs: ArrayLike, key: ArrayLike) -> tuple[Array, Array]:
        keys = random.split(key, obs.shape[0] + 1)
        actions = jax.vmap(act, in_axes=(None, 0, 0))(actor_params, obs, keys[:-1])
        return actions, keys[-1]

    def action_log_prob(actor_params, obs: ArrayLike) -> Array:
        logits = actor_model.apply(actor_params, obs)
        return nn.log_softmax(logits)

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

        #loss += l2_init_regularization(critic_params, init_critic_params, alpha=0.001)

        return loss

    def update_critic(
            critic_training_state: TrainState,
            obs: ArrayLike,
            expected_values: ArrayLike
    ) -> tuple[TrainState, Array]:
        critic_params = critic_training_state.params
        loss, gradient = jax.value_and_grad(critic_loss)(critic_params, obs, expected_values)
        return critic_training_state.apply_gradients(grads=gradient), loss

    def actor_loss(actor_params, states, actions, advantages, actor_l2_regularization, entropy_regularization) -> tuple[Array, tuple[Array, Array]]:
        action_probs = jax.vmap(action_log_prob, (None, 0))(actor_params, states)

        selected_action_prob = action_probs[jnp.arange(action_probs.shape[0]), actions]
        loss = -jnp.mean(selected_action_prob * advantages)

        l2_loss = l2_init_regularization(actor_params, init_actor_params, alpha=actor_l2_regularization)
        e_loss = entropy_regularization * jnp.mean(jax.vmap(entropy_loss)(action_probs))

        if debug:
            jax.debug.print("actor_loss - actions: {}", actions)
            jax.debug.print("actor_loss - action_probs: {}", action_probs)
            jax.debug.print("actor_loss - selected_action_prob: {}", selected_action_prob)
            # jax.debug.print("actor_loss - l2_loss: {}", l2_loss)
            jax.debug.print("actor_loss - e_loss: {}", e_loss)

        return loss + e_loss, (l2_loss, e_loss)

    def update_actor(actor_training_state: TrainState, states, actions, advantages, actor_l2_regularization, entropy_regularization) -> tuple[TrainState, Array, Array, Array]:
        (loss, (l2_loss, e_loss)), gradient = jax.value_and_grad(actor_loss, has_aux=True)(actor_training_state.params, states, actions, advantages, actor_l2_regularization, entropy_regularization)
        return actor_training_state.apply_gradients(grads=gradient), loss, l2_loss, e_loss

    def vectorized_train_step(
            params: Params,
            obs: ArrayLike,
            action: ArrayLike,
            next_obs: ArrayLike,
            rewards: ArrayLike,
            done: ArrayLike,
            key: ArrayLike
    ) -> tuple[Params, Array, Array, Metrics]:
        discount = params['discount']
        actor_l2_regularization = params['actor_l2_regularization']
        entropy_regularization = params['entropy_regularization']
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
        actor_training_state, a_loss, l2_loss, e_loss = update_actor(actor_training_state, obs, action, advantages, actor_l2_regularization, entropy_regularization)

        # set the importance back to 1 if it's the end of an episode
        importance *= discount
        importance = jnp.maximum(importance, done)

        output_params: Params = {
            'discount': discount,
            'actor_l2_regularization': actor_l2_regularization,
            'entropy_regularization': entropy_regularization,
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
            'entropy_loss': e_loss,
            'l2_loss': l2_loss,
        }

        return output_params, actions, key, metrics

    return (jax.jit(vectorized_train_step, donate_argnames=['params', 'action', 'key']),
            jax.jit(vectorized_act, donate_argnames=['key']),
            jax.jit(act))
