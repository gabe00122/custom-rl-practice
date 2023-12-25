from pathlib import Path
from .initialize import RunSettings
from .train import train


def base_settings() -> RunSettings:
    return {
        'seed': 123,
        'total_steps': 1_000_000,
        'env_name': 'LunarLander-v2',
        'env_num': 1,
        'discount': 0.99,
        'actor_hidden_layers': [64, 64],
        'critic_hidden_layers': [64, 64],
        'actor_last_layer_scale': 0.01,
        'critic_last_layer_scale': 0.5,
        'actor_learning_rate': 0.00025,
        'critic_learning_rate': 0.001,
        'actor_adam_beta': 0.99,
        'critic_adam_beta': 0.99,
        'actor_l2_regularization': 0.0001,
    }


def main():
    run_dir = Path('./run').absolute()
    for i in range(4):
        settings = base_settings()
        settings['seed'] = i
        if i < 2:
            settings['env_num'] = 1
        else:
            settings['env_num'] = 4

        train(settings, run_dir / f'0{i}')


if __name__ == "__main__":
    main()
