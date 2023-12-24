import gymnasium as gym
from ..networks.mlp import Mlp
from ..policy_gradient.actor_critic_v3 import actor_critic_v3
from jax import random, numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from flax.training.train_state import TrainState
import optax


def main():
    env_name = 'CartPole-v1'
    num_envs = 4
    env = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode="sync")
    action_space = env.single_action_space.n
    state_space = env.single_observation_space.shape[0]

    key = random.PRNGKey(53245)

    actor_model = Mlp(features=[64, 64, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=0.1)

    state_vector = jnp.zeros((state_space,), dtype=jnp.float32)

    key, actor_key, critic_key = random.split(key, 3)
    actor_params = actor_model.init(actor_key, state_vector)
    critic_params = critic_model.init(critic_key, state_vector)

    beta = 0.99
    params = {
        'discount': 0.999,
        'actor_training_state': TrainState.create(apply_fn=actor_model.apply, params=actor_params,
                                                  tx=optax.adam(0.00025 * 8,
                                                                b1=beta, b2=beta)),
        'critic_training_state': TrainState.create(apply_fn=critic_model.apply, params=critic_params,
                                                   tx=optax.adam(0.0005 * 8, b1=beta,
                                                                 b2=beta)),
    }

    vectorized_train_step, vectorized_act, act = actor_critic_v3(actor_model, critic_model)

    total_steps = 40000

    obs, info = env.reset(seed=42)
    obs = jnp.array(obs)

    actions, key = vectorized_act(actor_params, obs, key)
    importance = jnp.ones((num_envs,))

    rewards = np.zeros((total_steps,))
    state_values = np.zeros((total_steps,))
    td_errors = np.zeros((total_steps,))
    actor_loss = np.zeros((total_steps,))
    critic_loss = np.zeros((total_steps,))

    # metrics
    finished_episode_rewards = np.zeros((num_envs,))
    episode_rewards = np.zeros((num_envs,))
    episode_zeros = np.zeros((num_envs,))

    max_discounted_reward = 500 * (params['discount'] ** 500)

    with tqdm(range(total_steps), unit='steps') as tsteps:
        for step in tsteps:
            tsteps.set_description(f"Step {step}")

            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            reward /= max_discounted_reward
            next_obs = jnp.array(next_obs, dtype=jnp.float32)

            done = np.logical_or(terminated, truncated)
            params, importance, actions, key, metrics = vectorized_train_step(params, obs, actions, next_obs, reward,
                                                                              done, importance, key)

            obs = next_obs

            episode_rewards += reward
            finished_episode_rewards = np.where(done, episode_rewards, finished_episode_rewards)
            episode_rewards = np.where(done, episode_zeros, episode_rewards)

            rewards[step] = np.mean(finished_episode_rewards)
            state_values[step] = metrics['state_value']
            td_errors[step] = metrics['td_error']
            actor_loss[step] = metrics['actor_loss']
            critic_loss[step] = metrics['critic_loss']

            tsteps.set_postfix(reward=np.mean(finished_episode_rewards).item(),
                               state_values=metrics['state_value'].item(),
                               td_error=metrics['td_error'].item(),
                               )

    # let's see it in action
    actor_params = params['actor_training_state'].params
    env = gym.make('CartPole-v1', render_mode='human')

    for _ in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            key, act_key = random.split(key)
            action = act(actor_params, obs, act_key)
            obs, _, truncated, terminated, _ = env.step(action.item())
            done = truncated or terminated

    dataframe = pd.DataFrame({
        # 'reward': rewards / 500,
        # 'state_value': state_values / 500,
        'td_error': td_errors,
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
    })
    dataframe.to_csv('results.csv')
    dataframe.rolling(1000).mean().plot()
    plt.show()


if __name__ == "__main__":
    main()
