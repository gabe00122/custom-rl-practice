import gymnasium as gym
import numpy as np
from typing_extensions import TypedDict
from hyper_params import HyperParams

Policy = TypedDict('Policy', {'Q1': np.ndarray, 'Q2': np.ndarray})


def create_policy(state_space: int, action_space: int) -> Policy:
    return {
        'Q1': np.zeros((state_space, action_space)),
        'Q2': np.zeros((state_space, action_space)),
    }


def act(q: np.ndarray, params: HyperParams, state: int) -> int:
    if np.random.uniform() < params['exploration']:
        action_space = q.shape[1]
        return np.random.randint(action_space)
    else:
        return np.argmax(q[state])


def greedy_act(q: np.ndarray, params: HyperParams, state: int) -> int:
    return np.argmax(q[state])


def train_episode(policy: Policy, params: HyperParams, env: gym.Env):
    q1 = policy['Q1']
    q2 = policy['Q2']
    if np.random.uniform() > 0.5:
        q1, q2 = q2, q1

    rewards = 0
    learning_rate = params['learning_rate']
    discount = params['discount']
    done = False

    obs, _ = env.reset()

    while not done:
        action = act(q1, params, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        action_index = np.argmax(q2[next_obs])
        td_error = reward + (discount * q1[next_obs, action_index] - q1[obs, action])
        q1[obs, action] += learning_rate * td_error

        done = terminated or truncated
        obs = next_obs
        if np.random.uniform() > 0.5:
            q1, q2 = q2, q1

        rewards += reward

    return rewards


def eval_episode(policy: Policy, params: HyperParams, env: gym.Env):
    q1 = policy['Q1']
    q2 = policy['Q2']
    if np.random.uniform() > 0.5:
        q1, q2 = q2, q1

    rewards = 0
    done = False

    obs, _ = env.reset()

    while not done:
        action = greedy_act(q1, params, obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        if np.random.uniform() > 0.5:
            q1, q2 = q2, q1

        rewards += reward

    return rewards


def main():
    env = gym.make("FrozenLake-v1")
    log_every = 100
    reward_sum = 9

    hyper_params: HyperParams = {
        'exploration': 0.1,
        'discount': 0.90,
        'learning_rate': 0.1,
    }

    policy = create_policy(env.observation_space.n, env.action_space.n)
    for i in range(500000):
        reward_sum += train_episode(policy, hyper_params, env)
        if i % log_every == log_every - 1:
            print(reward_sum / log_every)
            reward_sum = 0

    env = gym.make("FrozenLake-v1", render_mode="human")
    train_episode(policy, hyper_params, env)


if __name__ == '__main__':
    main()
