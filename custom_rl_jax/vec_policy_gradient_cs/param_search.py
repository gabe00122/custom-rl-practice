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
        total_steps=500000,
        env_name="CartPole-v1",
        env_num=1,
        discount=0.99, #trail.suggest_float('discount', 0.99, 1.0),
        root_hidden_layers=[64],
        actor_hidden_layers=[64, 64],
        critic_hidden_layers=[64, 64],
        actor_last_layer_scale=0.01,
        critic_last_layer_scale=1.0,
        optimizer="adam",
        learning_rate=0.0001, #trail.suggest_float('learning_rate', 0.00001, 0.001),
        adam_beta=0.97, #trail.suggest_float('adam_beta', 0.9, 0.9999),
        weight_decay=0.0, #trail.suggest_float('weight_decay', 0.0, 0.01),
        clip_norm=50.0, #trail.suggest_float('clip_norm', 0.0, 50.0),
        actor_coef=0.25, #trail.suggest_float('actor_coef', 0.1, 1.0),
        critic_coef=1.0, #trail.suggest_float('critic_coef', 0.5, 1.0),
        entropy_coef=0.0, #trail.suggest_float('entropy_coef', 0.0, 0.1),
    )


base_path = Path("./search").absolute()
# trial_num = 0


def objective(trial: optuna.Trial):
    # trial_num += 1
    settings = sample_settings(trial)
    result = train(settings, base_path / "1")
    jax.clear_caches()
    return result


def main():
    sampler = TPESampler(n_startup_trials=5)

    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="cartpole"
    )
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
    #main()
    settings = sample_settings(None)
    result = train(settings, base_path / "1")
