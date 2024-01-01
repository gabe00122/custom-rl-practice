from pathlib import Path
from .run_settings import RunSettings
from .train_gymnax import train


def main():
    settings = RunSettings(
        git_hash='tbd',
        seed=57584,
        total_steps=100_000,
        env_name='CartPole-v1',
        env_num=10000,
        discount=0.99,
        actor_hidden_layers=[128, 128, 128],
        critic_hidden_layers=[128, 128, 128],
        actor_last_layer_scale=0.01,
        critic_last_layer_scale=1.0,
        actor_learning_rate=2 ** -10,
        critic_learning_rate=2 ** -9,
        actor_adam_beta=0.97,
        critic_adam_beta=0.97,

        critic_regularization_type='l2-init',
        critic_regularization_alpha=0.001,
        actor_regularization_type='l2-init',
        actor_regularization_alpha=0.001,
        actor_entropy_regularization=0,
    )

    train(settings, Path('./run-cart-pole-jax'))


if __name__ == "__main__":
    main()
