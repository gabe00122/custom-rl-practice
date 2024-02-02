import jax
from jax import Array, random, numpy as jnp
from jax.typing import ArrayLike
from flax import linen as nn, struct
from flax.core.frozen_dict import FrozenDict, Mapping
from flax.struct import PyTreeNode
import optax
from typing import Any, NamedTuple
from .metrics.metrics_type import Metrics
from .regularization import entropy_loss

type ModelParams = FrozenDict[str, Mapping[str, Any]] | dict[str, Any]


class TrainingState(NamedTuple):
    model_params: ModelParams
    opt_state: optax.OptState


class ActorCritic(PyTreeNode):
    action_space: int = struct.field(pytree_node=False)
    observation_space: int = struct.field(pytree_node=False)

    model: nn.Module = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    discount: float = struct.field(pytree_node=False, default=0.99)

    actor_coef: float = struct.field(pytree_node=False, default=0.5)
    critic_coef: float = struct.field(pytree_node=False, default=1.0)
    entropy_coef: float = struct.field(pytree_node=False, default=0.0)

    def init(
        self,
        key: ArrayLike,
    ) -> TrainingState:
        observation_dummy = jnp.zeros((self.observation_space,))
        model_params = self.model.init(key, observation_dummy)

        return TrainingState(
            model_params=model_params,
            opt_state=self.optimizer.init(model_params),
        )

    @jax.jit
    def act(self, params: TrainingState, obs: ArrayLike, key: ArrayLike) -> Array:
        logits, _ = self.model.apply(params.model_params, obs)
        return random.categorical(key, logits)

    @jax.jit
    def loss(
        self,
        model_params: ModelParams,
        obs: ArrayLike,
        actions: ArrayLike,
        rewards: ArrayLike,
        next_obs: ArrayLike,
        done: ArrayLike,
        importance: ArrayLike,
    ) -> tuple[Array, Metrics]:
        v_model = jax.vmap(self.model.apply, (None, 0), (0, 0))
        v_entropy_loss = jax.vmap(entropy_loss)
        v_log_softmax = jax.vmap(nn.log_softmax)

        action_logits, vf_values = v_model(model_params, obs)
        _, next_critic_values = v_model(jax.lax.stop_gradient(model_params), next_obs)

        returns = jnp.where(
            done,
            rewards,
            rewards + self.discount * next_critic_values,
        )

        td_error = returns - vf_values
        critic_loss = (td_error**2).mean()

        action_probs = v_log_softmax(action_logits)
        selected_action_prob = action_probs[jnp.arange(action_probs.shape[0]), actions]
        actor_loss = -jnp.mean(selected_action_prob * td_error * importance)

        entropy = jnp.mean(v_entropy_loss(action_probs))

        loss = self.actor_coef * actor_loss + critic_loss

        metrics: Metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "td_error": jnp.mean(td_error),
            "state_value": jnp.mean(returns),
            "entropy": entropy,
        }

        return loss, metrics

    @jax.jit
    def update_model(
        self,
        params: TrainingState,
        obs: ArrayLike,
        actions: ArrayLike,
        rewards: ArrayLike,
        next_obs: ArrayLike,
        done: ArrayLike,
        importance: ArrayLike,
    ) -> tuple[TrainingState, Metrics]:
        loss_fn = jax.value_and_grad(self.loss, has_aux=True)
        (loss, metrics), grad = loss_fn(params.model_params, obs, actions, rewards, next_obs, done, importance)

        updates, opt_state = self.optimizer.update(grad, params.opt_state, params.model_params)
        model_params = optax.apply_updates(params.model_params, updates)

        next_params = TrainingState(model_params, opt_state)

        return next_params, metrics

    @jax.jit
    def train_step(
        self,
        params: TrainingState,
        obs: ArrayLike,
        actions: ArrayLike,
        rewards: ArrayLike,
        next_obs: ArrayLike,
        done: ArrayLike,
        importance: ArrayLike,
    ) -> tuple[TrainingState, Metrics, Array]:
        assert obs.dtype == jnp.float32
        assert actions.dtype == jnp.int32
        assert rewards.dtype == jnp.float32
        assert next_obs.dtype == jnp.float32
        assert done.dtype == jnp.bool_
        assert importance.dtype == jnp.float32

        params, metrics = self.update_model(params, obs, actions, rewards, next_obs, done, importance)

        # set the importance back to 1 if it's the end of an episode
        importance = jnp.maximum(importance * self.discount, done)
        return params, metrics, importance
