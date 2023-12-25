from typing_extensions import TypedDict
from jax import numpy as jnp, random
from flax.training.train_state import TrainState
import flax.linen as nn
import optax
from ..networks.mlp import Mlp
from .actor_critic_v3 import Params


RunSettings = TypedDict('RunSettings', {
    'seed': int,
    'total_steps': int,
    'env_name': str,
    'env_num': int,

    'discount': float,
    'actor_hidden_layers': list[int],
    'critic_hidden_layers': list[int],
    'actor_last_layer_scale': float,
    'critic_last_layer_scale': float,

    'actor_learning_rate': float,
    'critic_learning_rate': float,
    'actor_adam_beta': float,
    'critic_adam_beta': float,

    'actor_l2_regularization': float,
})


def create_models(settings: RunSettings, action_space: int) -> tuple[nn.Module, nn.Module]:
    actor_model = Mlp(
        features=settings['actor_hidden_layers'] + [action_space],
        last_layer_scale=settings['actor_last_layer_scale'],
    )
    critic_model = Mlp(
        features=settings['critic_hidden_layers'] + [1],
        last_layer_scale=settings['critic_last_layer_scale'],
    )

    return actor_model, critic_model


def create_training_params(
        settings: RunSettings,
        actor_model: nn.Module,
        critic_model: nn.Module,
        state_space: int,
        key: random.KeyArray,
) -> tuple[Params, random.KeyArray]:
    state_input = jnp.zeros((state_space,))
    key, actor_key, critic_key = random.split(key, 3)

    actor_params = actor_model.init(actor_key, state_input)
    critic_params = critic_model.init(critic_key, state_input)

    actor_beta = settings['actor_adam_beta']
    actor_lr = settings['actor_learning_rate']
    actor_training_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(actor_lr, b1=actor_beta, b2=actor_beta)
    )

    critic_beta = settings['critic_adam_beta']
    critic_lr = settings['critic_learning_rate']
    critic_training_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(critic_lr, b1=critic_beta, b2=critic_beta)
    )

    return {
        'discount': settings['discount'],
        'actor_l2_regularization': settings['actor_l2_regularization'],
        'importance': jnp.ones((settings['env_num'],)),
        'actor_training_state': actor_training_state,
        'critic_training_state': critic_training_state,
    }, key
