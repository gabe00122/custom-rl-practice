from jax.typing import ArrayLike
from flax import linen as nn
import optax
from ..networks.mlp import Mlp
from .run_settings import RunSettings
from .actor_critic import ActorCritic


def create_actor_model(settings: RunSettings, action_space: int) -> nn.Module:
    return Mlp(
        features=settings['actor_hidden_layers'] + [action_space],
        last_layer_scale=settings['actor_last_layer_scale'],
    )


def create_critic_model(settings: RunSettings) -> nn.Module:
    return Mlp(
        features=settings['critic_hidden_layers'] + [1],
        last_layer_scale=settings['critic_last_layer_scale'],
    )


def create_actor_critic(settings: RunSettings, action_space: int, observation_space: int) -> ActorCritic:
    actor_model = create_actor_model(settings, action_space)
    actor_optimizer = optax.adam(settings['actor_learning_rate'], b1=settings['actor_adam_beta'], b2=settings['actor_adam_beta'])
    critic_model = create_critic_model(settings)
    critic_optimizer = optax.adam(settings['critic_learning_rate'], b1=settings['critic_adam_beta'], b2=settings['critic_adam_beta'])

    actor_critic = ActorCritic(
        action_space=action_space,
        observation_space=observation_space,
        actor_model=actor_model,
        actor_optimizer=actor_optimizer,
        critic_model=critic_model,
        critic_optimizer=critic_optimizer,
        discount=settings['discount'],
        actor_regularization_alpha=settings['actor_regularization_alpha'],
        actor_entropy_regularization=settings['actor_entropy_regularization'],
        critic_regularization_alpha=settings['actor_regularization_alpha'],
    )
    return actor_critic
