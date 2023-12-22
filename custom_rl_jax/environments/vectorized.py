import gymnasium as gym
from ..networks.mlp import Mlp
from ..policy_gradient.actor_critic_v3 import actor_critic_v3
from jax import random, numpy as jnp

from tqdm import tqdm
from pathlib import Path
import shutil
import orbax.checkpoint as ocp

from flax.training.train_state import TrainState
import optax


def main():
    env_name = 'CartPole-v1'
    num_envs = 8
    env = gym.make_vec(env_name, num_envs=num_envs)
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
        'discount': 0.999,
        'actor_training_state': TrainState.create(apply_fn=actor_model.apply, params=actor_params,
                                                  tx=optax.adam(0.00025,
                                                                b1=beta, b2=beta)),
        'critic_training_state': TrainState.create(apply_fn=critic_model.apply, params=critic_params,
                                                   tx=optax.adam(0.001, b1=beta,
                                                                 b2=beta)),
    }

    train_episode, act, vectorized_act = actor_critic_v3(actor_model, critic_model, num_envs)

    total_episodes = 4000

    obs, info = env.reset(seed=42)

    with tqdm(range(total_episodes), unit='episode') as tepisode:
        for episode in tepisode:
            tepisode.set_description(f"Episode {episode}")

            vectorized_act, key = vectorized_act(actor_params, obs, key)
            print(vectorized_act)

            obs, reward, terminated, truncated, info = env.step(vectorized_act)

            # metrics, params, key = train_episode(params, key, env)
            # rewards[episode] = metrics['reward']
            # state_values[episode] = metrics['state_value']
            # td_errors[episode] = metrics['td_error']

            # tepisode.set_postfix(reward=metrics['reward'], state=round(metrics['state_value']))



if __name__ == "__main__":
    main()
