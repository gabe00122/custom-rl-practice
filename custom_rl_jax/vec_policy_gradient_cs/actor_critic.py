import jax
from jax import Array, random, numpy as jnp
from jax.typing import ArrayLike
from flax import linen as nn, struct
from flax.core.frozen_dict import FrozenDict, Mapping
from flax.struct import PyTreeNode
import optax
from typing import Any, TypedDict
from .regularization import l2_regularization, l2_init_regularization, entropy_loss

type ModelParams = FrozenDict[str, Mapping[str, Any]] | dict[str, Any]


Metrics = TypedDict(
    "Metrics",
    {
        "state_value": Array,
        "td_error": Array,
        "actor_loss": Array,
        "critic_loss": Array,
        "entropy": Array,
        "actor_reg_loss": Array,
        "critic_reg_loss": Array,
    },
)


class TrainingState(PyTreeNode):
    init_actor_params: ModelParams
    actor_params: ModelParams
    actor_opt_state: optax.OptState

    init_critic_params: ModelParams
    critic_params: ModelParams
    critic_opt_state: optax.OptState


class ActorCritic(PyTreeNode):
    action_space: int = struct.field(pytree_node=False)
    observation_space: int = struct.field(pytree_node=False)

    actor_model: nn.Module = struct.field(pytree_node=False)
    actor_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_model: nn.Module = struct.field(pytree_node=False)
    critic_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    discount: float = struct.field(pytree_node=False, default=0.99)

    actor_regularization_type: str = struct.field(pytree_node=False, default="l2_init")
    actor_regularization_alpha: float = struct.field(pytree_node=False, default=0.001)
    actor_entropy_regularization: float = struct.field(pytree_node=False, default=0.0)

    critic_regularization_type: str = struct.field(pytree_node=False, default="l2_init")
    critic_regularization_alpha: float = struct.field(pytree_node=False, default=0.001)

    def init(
        self,
        key: ArrayLike,
    ) -> TrainingState:
        actor_key, critic_key = random.split(key)
        observation_dummy = jnp.zeros((self.observation_space,))

        actor_params = self.actor_model.init(actor_key, observation_dummy)
        critic_params = self.critic_model.init(critic_key, observation_dummy)
        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)

        return TrainingState(
            init_actor_params=actor_params,
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            init_critic_params=critic_params,
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
        )

    @jax.jit
    def act(self, params: TrainingState, obs: ArrayLike, key: ArrayLike) -> Array:
        logits = self.actor_model.apply(params.actor_params, obs)
        return random.categorical(key, logits)

    @jax.jit
    def action_log_prob(self, actor_params: ModelParams, obs: ArrayLike) -> Array:
        logits = self.actor_model.apply(actor_params, obs)
        return nn.log_softmax(logits)

    @jax.jit
    def state_values(self, critic_params: ModelParams, obs: ArrayLike) -> Array:
        return self.critic_model.apply(critic_params, obs).reshape(())

    @jax.jit
    def critic_loss(
        self, critic_params: ModelParams, init_critic_params: ModelParams, obs: ArrayLike, expected_values: ArrayLike
    ) -> tuple[Array, tuple[Array, Array]]:
        state_values_rng = jax.vmap(self.state_values, (None, 0))

        critic_values = state_values_rng(critic_params, obs)
        td_error = expected_values - critic_values
        loss = (td_error**2).mean()

        reg_loss = 0.0
        if self.critic_regularization_type == "l2_init":
            reg_loss = l2_init_regularization(
                critic_params,
                init_critic_params,
                alpha=self.critic_regularization_alpha,
            )
        elif self.critic_regularization_type == "l2":
            reg_loss = l2_regularization(critic_params, alpha=self.critic_regularization_alpha)

        loss += reg_loss

        return loss, (td_error, reg_loss)

    @jax.jit
    def update_critic(
        self, params: TrainingState, obs: ArrayLike, expected_values: ArrayLike
    ) -> tuple[TrainingState, Array, Array, Array]:
        (loss, (td_error, reg_loss)), gradient = jax.value_and_grad(self.critic_loss, has_aux=True)(
            params.critic_params, params.init_critic_params, obs, expected_values
        )
        updates, updated_opt_state = self.critic_optimizer.update(
            gradient, params.critic_opt_state, params.critic_params
        )
        updated_params = optax.apply_updates(params.critic_params, updates)

        return (
            params.replace(
                critic_params=updated_params,
                critic_opt_state=updated_opt_state,
            ),
            td_error,
            loss,
            reg_loss,
        )

    @jax.jit
    def actor_loss(
        self,
        actor_params: ModelParams,
        init_actor_params: ModelParams,
        states: ArrayLike,
        actions: ArrayLike,
        td_error: ArrayLike,
    ) -> tuple[Array, tuple[Array, Array]]:
        action_log_prob_rng = jax.vmap(self.action_log_prob, (None, 0))
        entropy_loss_rng = jax.vmap(entropy_loss)

        action_probs = action_log_prob_rng(actor_params, states)
        entropy = jnp.mean(entropy_loss_rng(action_probs))

        selected_action_prob = action_probs[jnp.arange(action_probs.shape[0]), actions]
        loss = -jnp.mean(selected_action_prob * td_error)

        reg_loss = 0.0
        if self.actor_regularization_type == "l2_init":
            reg_loss = l2_init_regularization(actor_params, init_actor_params, alpha=self.actor_regularization_alpha)
        elif self.actor_regularization_type == "l2":
            reg_loss = l2_regularization(actor_params, alpha=self.actor_regularization_alpha)

        loss += reg_loss

        if self.actor_entropy_regularization > 0.0:
            loss -= self.actor_entropy_regularization * entropy

        return loss, (reg_loss, entropy)

    @jax.jit
    def update_actor(
        self, params: TrainingState, states: ArrayLike, actions: ArrayLike, td_error: ArrayLike
    ) -> tuple[TrainingState, Array, Array, Array]:
        (loss, (reg_loss, entropy)), gradient = jax.value_and_grad(self.actor_loss, has_aux=True)(
            params.actor_params, params.init_actor_params, states, actions, td_error
        )
        updates, updated_opt_state = self.actor_optimizer.update(gradient, params.actor_opt_state, params.actor_params)
        updated_params = optax.apply_updates(params.actor_params, updates)
        return (
            params.replace(
                actor_params=updated_params,
                actor_opt_state=updated_opt_state,
            ),
            loss,
            reg_loss,
            entropy,
        )

    @jax.jit
    def train_step(
        self,
        params: TrainingState,
        obs: ArrayLike,
        action: ArrayLike,
        rewards: ArrayLike,
        next_obs: ArrayLike,
        done: ArrayLike,
        importance: ArrayLike,
    ) -> tuple[TrainingState, Metrics, Array]:
        state_value_rng = jax.vmap(self.state_values, (None, 0))

        expected_values = jnp.where(
            done,
            rewards,
            rewards + self.discount * jax.lax.stop_gradient(state_value_rng(params.critic_params, next_obs)),
        )

        params, td_error, c_loss, critic_reg_loss = self.update_critic(params, obs, expected_values)
        params, a_loss, actor_reg_loss, entropy = self.update_actor(params, obs, action, td_error * importance)

        # set the importance back to 1 if it's the end of an episode
        importance = jnp.maximum(importance * self.discount, done)

        metrics: Metrics = {
            "actor_loss": a_loss,
            "critic_loss": c_loss,
            "td_error": jnp.mean(td_error),
            "state_value": jnp.mean(expected_values),
            "entropy": entropy,
            "actor_reg_loss": actor_reg_loss,
            "critic_reg_loss": critic_reg_loss,
        }

        return params, metrics, importance
