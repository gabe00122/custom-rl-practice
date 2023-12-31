import json
from typing import TypedDict

RunSettings = TypedDict('RunSettings', {
    'git_hash': str,
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

    'critic_regularization_type': str,
    'critic_regularization_alpha': float,

    'actor_regularization_type': str,
    'actor_regularization_alpha': float,
    'actor_entropy_regularization': float,
})


def save_settings(path, settings):
    with open(path, 'w') as file:
        json.dump(settings, file, indent=2)


def load_settings(path):
    with open(path, 'r') as file:
        settings = json.load(file)
    return settings
