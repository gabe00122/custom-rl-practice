from pathlib import Path
from .settings import RunSettings
from .train import train


def base_settings() -> RunSettings:
    return {
        'seed': 666,
        'total_steps': 1_000_000,
        'env_name': 'LunarLander-v2',
        'env_num': 8,
        'discount': 0.997,
        'actor_hidden_layers': [64, 64],
        'critic_hidden_layers': [64, 64],
        'actor_last_layer_scale': 0.01,
        'critic_last_layer_scale': 0.5,
        'actor_learning_rate': 2 ** -10,
        'critic_learning_rate': 2 ** -9,
        'actor_adam_beta': 0.99,
        'critic_adam_beta': 0.99,
        'actor_l2_regularization': 0.001,
        'entropy_regularization': 0.00001,
    }


def main():
    run_dir = Path('./run-lander-997').absolute()
    #for i in range(4):
    settings = base_settings()
    # settings['seed'] = i
    # settings['env_num'] = i + 1

    train(settings, run_dir)


if __name__ == "__main__":
    main()
