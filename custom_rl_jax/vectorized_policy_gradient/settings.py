from typing import TypedDict


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
    'entropy_regularization': float,
})