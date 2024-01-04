import math
from pathlib import Path
from .settings import RunSettings
from .train import train


def scale(base, width, depth):
    return base / math.sqrt(width * depth)


def base_settings() -> RunSettings:
    return {
        'seed': 57584,
        'total_steps': 500_000,
        'env_name': 'LunarLander-v2',
        'env_num': 64,
        'discount': 0.99,
        'actor_hidden_layers': [64, 64],
        'critic_hidden_layers': [64, 64],
        'actor_last_layer_scale': 0.01,
        'critic_last_layer_scale': 1.0,
        'actor_learning_rate': 2 ** -10,
        'critic_learning_rate': 2 ** -9,
        'actor_adam_beta': 0.97,
        'critic_adam_beta': 0.97,
        'actor_l2_regularization': 0.001,
        'entropy_regularization': 0.000005,
    }


def main():
    run_dir = Path('./adamw-l2').absolute()
    #for i in range(4):
    settings = base_settings()
    # settings['seed'] = i
    # settings['env_num'] = i + 1

    train(settings, run_dir)


if __name__ == "__main__":
    main()
