import optuna
from optuna.samplers import TPESampler
import jax
from pathlib import Path
import random
from .run_settings import RunSettings
from .train_gymnax import train


def sample_settings(trail: optuna.Trial):
    return RunSettings(
        git_hash="tbd",
        seed=random.randint(0, 100_000),
        total_steps=50000,
        env_name="CartPole-v1",
        env_num=160,
        discount=trail.suggest_float('discount', 0.99, 1.0),
        actor_hidden_layers=[64, 64],
        critic_hidden_layers=[64, 64],
        actor_last_layer_scale=0.01,
        critic_last_layer_scale=1.0,
        actor_learning_rate=trail.suggest_float('actor_learning_rate', 0.00001, 0.001),
        critic_learning_rate=trail.suggest_float('critic_learning_rate', 0.00001, 0.001),
        actor_adam_beta=trail.suggest_float('actor_adam_beta', 0.9, 0.9999),
        critic_adam_beta=trail.suggest_float('critic_adam_beta', 0.9, 0.9999),
        critic_regularization_type="l2-init",
        critic_regularization_alpha=0,  # 0.0001,
        actor_regularization_type="l2-init",
        actor_regularization_alpha=0,  # 0.001,
        actor_entropy_regularization=0,  # 0.000005,

        actor_weight_decay=trail.suggest_float('actor_weight_decay', 0.0, 0.01),
        critic_weight_decay=trail.suggest_float('critic_weight_decay', 0.0, 0.01),
        actor_clip_norm=trail.suggest_float('actor_clip_norm', 0.0, 50.0),
        critic_clip_norm=trail.suggest_float('critic_clip_norm', 0.0, 50.0),
    )


base_path = Path("./search").absolute()
trial_num = 0


def objective(trial: optuna.Trial):
    trial_num += 1
    settings = sample_settings(trial)
    result = train(settings, base_path / f"{trial_num}")
    jax.clear_caches()
    return result


def main():
    sampler = TPESampler(n_startup_trials=5)

    study = optuna.create_study(sampler=sampler, direction="maximize")
    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
