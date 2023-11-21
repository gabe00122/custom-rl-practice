import gymnasium as gym
import numpy as np
from typing_extensions import TypedDict
from hyper_params import HyperParams

Policy = TypedDict('Policy', {'Q': np.ndarray})


def create_policy(state_space: int, action_space: int) -> Policy:
    return {
        'Q': np.zeros((state_space, action_space))
    }


def act(q: np.ndarray, params: HyperParams, state: int) -> int:
    if np.random.uniform() < params['exploration']:
        action_space = q.shape[1]
        return np.random.randint(action_space)
    else:
        return np.argmax(q[state])


def train_episode(policy: Policy, params: HyperParams, env: gym.Env):
    rewards = 0
    q = policy['Q']
    done = False
    exploration = params['exploration']
    discount = params['discount']
    learning_rate = params['learning_rate']

    obs, _ = env.reset()

    while not done:
        action = act(q, params, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        greedy_return = np.max(q[next_obs])
        exploratory_return = np.mean(q[next_obs])
        expected_return = (greedy_return * (1.0 - exploration)) + (exploratory_return * exploration)

        td_error = reward + (discount * expected_return - q[obs, action])
        q[obs, action] += learning_rate * td_error

        done = terminated or truncated
        obs = next_obs
        rewards += reward

    return rewards


def main():
    env = gym.make("FrozenLake-v1")
    log_every = 100
    reward_sum = 9

    hyper_params: HyperParams = {
        'exploration': 0.1,
        'discount': 0.99,
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
