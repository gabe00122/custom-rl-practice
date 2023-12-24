import math

import gymnasium as gym
from ..networks.mlp import Mlp
from ..policy_gradient.actor_critic_v3 import actor_critic_v3
from jax import random, numpy as jnp
import numpy as np

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
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)

    state_vector = jnp.zeros((state_space,), dtype=jnp.float32)

    key, actor_key, critic_key = random.split(key, 3)
    actor_params = actor_model.init(actor_key, state_vector)
    critic_params = critic_model.init(critic_key, state_vector)

    beta = 0.99
    params = {
        'discount': 0.99,
        'actor_training_state': TrainState.create(apply_fn=actor_model.apply, params=actor_params,
                                                  tx=optax.adam(0.00025,
                                                                b1=beta, b2=beta)),
        'critic_training_state': TrainState.create(apply_fn=critic_model.apply, params=critic_params,
                                                   tx=optax.adam(0.0005, b1=beta,
                                                                 b2=beta)),
    }

    vectorized_train_step, vectorized_act, act = actor_critic_v3(actor_model, critic_model, num_envs)

    total_steps = 40000

    obs, info = env.reset(seed=42)
    obs = jnp.array(obs)

    actions, key = vectorized_act(actor_params, obs, key)
    importance = jnp.ones((num_envs,))

    # metrics
    episode_rewards = np.zeros((num_envs,))
    episode_zeros = np.zeros((num_envs,))

    with tqdm(range(total_steps), unit='steps') as tsteps:
        for step in tsteps:
            tsteps.set_description(f"Step {step}")

            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            next_obs = jnp.array(next_obs, dtype=jnp.float32)

            done = np.logical_or(terminated, truncated)
            params, importance, actions, key, metrics = vectorized_train_step(params, obs, actions, next_obs, reward,
                                                                              done, importance, key)

            obs = next_obs

            episode_rewards += reward
            tsteps.set_postfix(reward=np.mean(episode_rewards).item(),
                               td_error=metrics['td_error'].item(),
                               )

            episode_rewards = np.where(done, episode_zeros, episode_rewards)
            # metrics, params, key = train_episode(params, key, env)
            # rewards[episode] = metrics['reward']
            # state_values[episode] = metrics['state_value']
            # td_errors[episode] = metrics['td_error']
            # print(np.mean(episode_rewards))

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


if __name__ == "__main__":
    main()
