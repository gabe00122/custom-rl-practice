from flax import linen as nn
import optax
from ..networks.mlp import Mlp
from ..networks.actor_critic_model import ActorCriticModel
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
    root_model = Mlp(features=settings["root_hidden_layers"])
    actor_model = create_actor_model(settings, action_space)
    critic_model = create_critic_model(settings)

    actor_critic_model = ActorCriticModel(root=root_model, actor=actor_model, critic=critic_model)

    optimizer = optax.chain(
        optax.clip_by_global_norm(settings["clip_norm"]),
        optax.adamw(
            optax.linear_schedule(settings["learning_rate"], 0, settings['total_steps']),
            b1=settings["adam_beta"],
            b2=settings["adam_beta"],
            weight_decay=settings["weight_decay"]
        ),
    )

    actor_critic = ActorCritic(
        action_space=action_space,
        observation_space=observation_space,
        model=actor_critic_model,
        optimizer=optimizer,
        discount=settings["discount"],
        actor_coef=settings["actor_coef"],
        critic_coef=settings["critic_coef"],
        entropy_coef=settings["entropy_coef"],
    )
    return actor_critic
