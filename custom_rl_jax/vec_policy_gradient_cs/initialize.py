from flax import linen as nn
import optax
from ..networks.mlp import Mlp
from .run_settings import RunSettings
from .actor_critic import ActorCritic


def create_actor_model(settings: RunSettings, action_space: int) -> nn.Module:
    return Mlp(
        features=settings["actor_hidden_layers"] + [action_space],
        last_layer_scale=settings["actor_last_layer_scale"],
    )


def create_critic_model(settings: RunSettings) -> nn.Module:
    return Mlp(
        features=settings["critic_hidden_layers"] + [1],
        last_layer_scale=settings["critic_last_layer_scale"],
    )


def create_actor_critic(settings: RunSettings, action_space: int, observation_space: int) -> ActorCritic:
    actor_model = create_actor_model(settings, action_space)
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(settings["actor_clip_norm"]),
        optax.adamw(settings["actor_learning_rate"], b1=0.9, b2=0.98, weight_decay=settings["critic_weight_decay"]),
    )
    critic_model = create_critic_model(settings)
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(settings["critic_clip_norm"]),
        optax.adamw(settings["critic_learning_rate"], b1=0.9, b2=0.98, weight_decay=settings["actor_weight_decay"]),
    )

    actor_critic = ActorCritic(
        action_space=action_space,
        observation_space=observation_space,
        actor_model=actor_model,
        actor_optimizer=actor_optimizer,
        critic_model=critic_model,
        critic_optimizer=critic_optimizer,
        discount=settings["discount"],
        actor_regularization_alpha=settings["actor_regularization_alpha"],
        actor_entropy_regularization=settings["actor_entropy_regularization"],
        critic_regularization_alpha=settings["actor_regularization_alpha"],
    )
    return actor_critic
