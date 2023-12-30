import jax
from jax import Array, random, numpy as jnp
from jax.typing import ArrayLike
from flax import linen as nn, struct
from flax.struct import PyTreeNode
import optax
from typing import Any, TypedDict
from functools import partial
from .regularization import l2_regularization, l2_init_regularization, entropy_loss

type ModelParams = dict[str, Any]


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


class ActorCritic(PyTreeNode):
    debug: bool = struct.field(pytree_node=False)
    actor_model: nn.Module = struct.field(pytree_node=False)
    critic_model: nn.Module = struct.field(pytree_node=False)

    init_actor_params: ModelParams = struct.field()
    init_critic_params: ModelParams = struct.field()
    actor_params: ModelParams = struct.field()
    critic_params: ModelParams = struct.field()

    actor_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    actor_opt_state: optax.OptState = struct.field()
    critic_opt_state: optax.OptState = struct.field()
    importance: Array = struct.field()

    discount: float = struct.field(default=0.99)

    critic_regularization_type: str = struct.field(pytree_node=False, default="l2_init")
    critic_regularization_alpha: float = struct.field(default=0.001)

    actor_regularization_type: str = struct.field(pytree_node=False, default="l2_init")
    actor_regularization_alpha: float = struct.field(default=0.001)
    actor_entropy_regularization: float = struct.field(pytree_node=False, default=0.000005)

    @classmethod
    def init(
        cls,
        actor_model: nn.Module,
        critic_model: nn.Module,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        obs_space: int,
        vec_num: int,
        key: ArrayLike,
    ) -> tuple["ActorCritic", Array]:
        key, actor_key, critic_key = random.split(key, 3)
        observation_dummy = jnp.zeros((obs_space,))

        actor_params = actor_model.init(actor_key, observation_dummy)
        critic_params = critic_model.init(critic_key, observation_dummy)
        actor_opt_state = actor_optimizer.init(actor_params)
        critic_opt_state = critic_optimizer.init(critic_params)

        importance = jnp.ones((vec_num,))

        return (
            cls(
                debug=False,
                actor_model=actor_model,
                critic_model=critic_model,
                init_actor_params=actor_params,
                init_critic_params=critic_params,
                actor_params=actor_params,
                critic_params=critic_params,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                actor_opt_state=actor_opt_state,
                critic_opt_state=critic_opt_state,
                importance=importance,
            ),
            key,
        )

    @jax.jit
    def act(self, obs: ArrayLike, key: ArrayLike) -> Array:
        logits = self.actor_model.apply(self.actor_params, obs)
        return random.categorical(key, logits)

    @jax.jit
    def vec_act(self, obs: ArrayLike, key: ArrayLike) -> tuple[Array, Array]:
        key, action_key = random.split(key)
        vec_actor_model = jax.vmap(self.actor_model.apply, in_axes=(None, 0))

        logits = vec_actor_model(self.actor_params, obs)
        actions = random.categorical(action_key, logits, axis=1)
        return actions, key

    @jax.jit
    def action_log_prob(self, actor_params: ModelParams, obs: ArrayLike) -> Array:
        logits = self.actor_model.apply(actor_params, obs)
        return nn.log_softmax(logits)

    @jax.jit
    def vec_state_values(self, critic_params: ModelParams, obs: ArrayLike) -> Array:
        vec_critic_model = jax.vmap(self.critic_model.apply, in_axes=(None, 0))
        return vec_critic_model(critic_params, obs).flatten()

    @jax.jit
    def critic_loss(
        self, critic_params: ModelParams, obs: ArrayLike, expected_values: ArrayLike
    ) -> tuple[Array, Array]:
        critic_values = self.vec_state_values(critic_params, obs)
        loss = ((expected_values - critic_values) ** 2).mean()

        reg_loss = 0.0
        if self.critic_regularization_type == "l2_init":
            reg_loss = l2_init_regularization(
                self.critic_params,
                self.init_critic_params,
                alpha=self.critic_regularization_alpha,
            )
        elif self.critic_regularization_type == "l2":
            reg_loss = l2_regularization(self.critic_params, alpha=self.critic_regularization_alpha)

        loss += reg_loss

        return loss, reg_loss

    @jax.jit
    def update_critic(
        self, obs: ArrayLike, expected_values: ArrayLike
    ) -> tuple[ModelParams, optax.OptState, Array, Array]:
        (loss, reg_loss), gradient = jax.value_and_grad(self.critic_loss, has_aux=True)(
            self.critic_params, obs, expected_values
        )
        updates, critic_opt_state = self.critic_optimizer.update(gradient, self.critic_opt_state, self.critic_params)
        updated_params = optax.apply_updates(self.critic_params, updates)

        return updated_params, critic_opt_state, loss, reg_loss

    @jax.jit
    def actor_loss(
        self,
        actor_params: ModelParams,
        states: ArrayLike,
        actions: ArrayLike,
        advantages: ArrayLike,
    ) -> tuple[Array, tuple[Array, Array]]:
        action_probs = jax.vmap(self.action_log_prob, (None, 0))(actor_params, states)
        selected_action_prob = action_probs[jnp.arange(action_probs.shape[0]), actions]

        loss = -jnp.mean(selected_action_prob * advantages)

        if self.actor_regularization_type == "l2_init":
            reg_loss = l2_init_regularization(
                actor_params, self.init_actor_params, alpha=self.actor_regularization_alpha
            )
        elif self.actor_regularization_type == "l2":
            reg_loss = l2_regularization(actor_params, alpha=self.actor_regularization_alpha)
        else:
            reg_loss = 0.0

        loss += reg_loss

        entropy = jnp.mean(jax.vmap(entropy_loss)(action_probs))
        if self.actor_entropy_regularization > 0.0:
            loss += self.actor_entropy_regularization * entropy

        return loss, (reg_loss, entropy)

    @jax.jit
    def update_actor(
        self, states: ArrayLike, actions: ArrayLike, advantages: ArrayLike
    ) -> tuple[ModelParams, optax.OptState, Array, Array, Array]:
        (loss, (reg_loss, entropy)), gradient = jax.value_and_grad(self.actor_loss, has_aux=True)(
            self.actor_params, states, actions, advantages
        )
        updates, actor_opt_state = self.actor_optimizer.update(gradient, self.actor_opt_state, self.actor_params)
        updated_params = optax.apply_updates(self.actor_params, updates)
        return updated_params, actor_opt_state, loss, reg_loss, entropy

    @partial(jax.jit, donate_argnums=(0,))
    def vec_train_step(
        self,
        obs: ArrayLike,
        action: ArrayLike,
        next_obs: ArrayLike,
        rewards: ArrayLike,
        done: ArrayLike,
        key: ArrayLike,
    ) -> tuple["ActorCritic", Array, Array, Metrics]:
        expected_values = self.vec_state_values(self.critic_params, obs)

        next_expected_values = jnp.where(
            done,
            rewards,
            rewards + self.discount * jax.lax.stop_gradient(self.vec_state_values(self.critic_params, next_obs)),
        )

        advantages = next_expected_values - expected_values
        advantages *= self.importance
        updated_critic_params, updated_critic_opt_state, c_loss, critic_reg_loss = self.update_critic(
            obs, next_expected_values
        )
        updated_action_params, updated_actor_opt_state, a_loss, actor_reg_loss, entropy = self.update_actor(
            obs, action, advantages
        )

        # set the importance back to 1 if it's the end of an episode
        updated_importance = jnp.maximum(self.importance * self.discount, done)

        updated_params = self.replace(
            critic_params=updated_critic_params,
            critic_opt_state=updated_critic_opt_state,
            actor_params=updated_action_params,
            actor_opt_state=updated_actor_opt_state,
            importance=updated_importance,
        )
        actions, key = updated_params.vec_act(next_obs, key)

        metrics: Metrics = {
            "actor_loss": a_loss,
            "critic_loss": c_loss,
            "td_error": jnp.mean(advantages),
            "state_value": jnp.mean(expected_values),
            "entropy": entropy,
            "actor_reg_loss": actor_reg_loss,
            "critic_reg_loss": critic_reg_loss,
        }

        return updated_params, actions, key, metrics
