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

from flax.training.train_state import TrainState
import optax


def main():
    env_name = 'LunarLander-v2'
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

    beta = 0.99
    # optax.adam(0.00035, b1=beta, b2=beta, eps=1e-5))
    params = {
        'discount': 0.99,
        'actor_training_state': TrainState.create(apply_fn=actor_model.apply, params=actor_params,
                                                  tx=optax.adam(0.00025,
                                                                b1=beta, b2=beta, eps=1e-5)),
        'critic_training_state': TrainState.create(apply_fn=critic_model.apply, params=critic_params,
                                                   tx=optax.adam(0.001, b1=beta,
                                                                 b2=beta, eps=1e-5)),
    }

    train_episode, _ = actor_critic_v2(actor_model, critic_model)

    total_episodes = 4000
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

            tepisode.set_postfix(reward=metrics['reward'], state=round(metrics['state_value']))

    # write hyperparameters
    run_path = Path("./run").absolute()
    # serialized_params = json.dumps(hyper_params)
    # with open(run_path / "params.json", 'w') as file:
    #     file.write(serialized_params)

    # save model
    checkpoint_path = Path(run_path)
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

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
