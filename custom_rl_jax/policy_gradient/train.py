import gymnasium as gym
from ..networks.mlp import Mlp
from .actor_critic_v2 import actor_critic_v2
from jax import random, numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import orbax.checkpoint as ocp
import json


def main():
    base_lr = 0.0002
    hyper_params = {
        'discount': 0.99,
        'actor_learning_rate': base_lr,
        'critic_learning_rate': base_lr * 2,
    }

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    key = random.PRNGKey(53245)

    actor_model = Mlp(features=[64, 64, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)

    state_vector = jnp.zeros((state_space,), dtype=jnp.float32)

    key, actor_key, critic_key = random.split(key, 3)
    actor_params = actor_model.init(actor_key, state_vector)
    critic_params = critic_model.init(critic_key, state_vector)
    params = {
        **hyper_params,
        'actor_params': actor_params,
        'critic_params': critic_params,
    }

    train_episode, _ = actor_critic_v2(actor_model, critic_model)

    total_episodes = 8000
    rewards = np.zeros((total_episodes,))
    state_values = np.zeros((total_episodes,))
    td_errors = np.zeros((total_episodes,))

    with tqdm(range(total_episodes), unit='episode') as tepisode:
        for episode in tepisode:
            tepisode.set_description(f"Episode {episode}")

            metrics, params, key = train_episode(params, key, env)
            rewards[episode] = metrics['reward']
            state_values[episode] = metrics['state_value']
            td_errors[episode] = metrics['td_error']

            tepisode.set_postfix(reward=metrics['reward'], state=metrics['state_value'])

    # write hyperparameters
    run_path = Path("./run").absolute()
    serialized_params = json.dumps(hyper_params)
    with open(run_path / "params.json", 'w') as file:
        file.write(serialized_params)

    # save model
    checkpoint_path = Path(run_path)
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    params = {
        'actor': actor_params,
        'critic': critic_params,
    }
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(checkpoint_path / "checkpoint", params)

    # save metrics
    data_frame = pd.DataFrame({
        'td-errors': td_errors,
        'state-values': state_values,
        'rewards': rewards,
    })

    data_path = Path(run_path).absolute()
    data_path.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(data_path / "log.cvs", index=False)


if __name__ == "__main__":
    main()
