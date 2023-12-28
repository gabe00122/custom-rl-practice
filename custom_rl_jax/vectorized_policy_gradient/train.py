import json
from pathlib import Path
import shutil
from jax import numpy as jnp, random
import numpy as np
import pandas as pd
import gymnasium as gym
import orbax.checkpoint as ocp
from tqdm import tqdm
from .initialize import RunSettings, create_models, create_training_params
from .actor_critic_v3 import actor_critic_v3


def train(settings: RunSettings, path: Path):
    key = random.PRNGKey(settings['seed'])

    env = gym.make_vec(settings['env_name'], num_envs=settings['env_num'], vectorization_mode="sync")
    action_space = env.single_action_space.n
    state_space = env.single_observation_space.shape[0]

    actor_model, critic_model = create_models(settings, action_space)
    params, key = create_training_params(settings, actor_model, critic_model, state_space, key)

    vectorized_train_step, vectorized_act, act = actor_critic_v3(settings, actor_model, critic_model)

    total_steps = settings['total_steps']
    num_envs = settings['env_num']

    obs, _ = env.reset(seed=settings['seed'])
    obs = jnp.asarray(obs)

    actions, key = vectorized_act(params['actor_training_state'].params, obs, key)

    # metrics
    rewards = np.zeros((total_steps,))
    state_values = np.zeros((total_steps,))
    td_errors = np.zeros((total_steps,))
    actor_loss = np.zeros((total_steps,))
    critic_loss = np.zeros((total_steps,))
    entropy_loss = np.zeros((total_steps,))
    l2_loss = np.zeros((total_steps,))

    # vectorized metrics
    finished_episode_rewards = np.zeros((num_envs,))
    episode_rewards = np.zeros((num_envs,))
    episode_zeros = np.zeros((num_envs,))
    mean_finished_episode_rewards = 0

    #max_discounted_reward = 500 * (params['discount'] ** 500)

    with tqdm(range(total_steps), unit='steps') as tsteps:
        for step in tsteps:
            tsteps.set_description(f"Step {step}")

            next_obs, reward, terminated, truncated, info = env.step(np.asarray(actions))
            #reward /= max_discounted_reward
            next_obs = jnp.asarray(next_obs, dtype=jnp.float32)

            done = np.logical_or(terminated, truncated)
            params, actions, key, metrics = vectorized_train_step(
                params,
                obs,
                actions,
                next_obs,
                reward,
                done,
                key
            )

            obs = next_obs

            episode_rewards += reward
            if np.any(done):
                finished_episode_rewards = np.where(done, episode_rewards, finished_episode_rewards)
                episode_rewards = np.where(done, episode_zeros, episode_rewards)

                mean_finished_episode_rewards = np.mean(finished_episode_rewards)

            rewards[step] = mean_finished_episode_rewards
            state_values[step] = metrics['state_value']
            td_errors[step] = metrics['td_error']
            actor_loss[step] = metrics['actor_loss']
            critic_loss[step] = metrics['critic_loss']
            entropy_loss[step] = metrics['entropy_loss']
            l2_loss[step] = metrics['l2_loss']

            tsteps.set_postfix(
                reward=f"{np.mean(finished_episode_rewards).item():.5f}",
                state_values=f"{metrics['state_value'].item():.2f}",
                td_error=f"{metrics['td_error'].item():.1f}",
            )

    env.close()

    create_directory(path)
    save_settings(path / "settings.json", settings)
    save_metrics(path / "metrics.csv", rewards, state_values, td_errors, actor_loss, critic_loss, entropy_loss, l2_loss)
    save_params(path / "params", params)


def create_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_settings(path, settings):
    with open(path, 'w') as file:
        json.dump(settings, file, indent=2)


def save_metrics(path, rewards, state_values, td_errors, actor_loss, critic_loss, entropy_loss, l2_loss):
    data_frame = pd.DataFrame({
        'td-errors': td_errors,
        'state-values': state_values,
        'rewards': rewards,
        'actor-loss': actor_loss,
        'critic-loss': critic_loss,
        'entropy-loss': entropy_loss,
        'l2-loss': l2_loss,
    })

    data_frame.to_csv(path, index=False)


def save_params(path, params):
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, params)

