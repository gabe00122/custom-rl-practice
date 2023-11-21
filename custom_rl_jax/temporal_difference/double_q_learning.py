import gymnasium as gym
import numpy as np
from typing_extensions import TypedDict

Policy = TypedDict('Policy', {'Q1': np.ndarray, 'Q2': np.ndarray, 'exploration': float, 'learning_rate': float, 'discount': float,
                              'state_space': int,
                              'action_space': int})


def create_policy(state_space: int, action_space: int, exploration: float, learning_rate: float,
                  discount: float) -> Policy:
    return {
        'Q1': np.zeros((state_space, action_space)),
        'Q2': np.zeros((state_space, action_space)),
        'exploration': exploration,
        'learning_rate': learning_rate,
        'discount': discount,
        'state_space': state_space,
        'action_space': action_space
    }


def act(policy: Policy, q: np.ndarray, state: int) -> int:
    if np.random.uniform() < policy['exploration']:
        return np.random.randint(policy['action_space'])
    else:
        return np.argmax(q[state])


def train_episode(policy: Policy, env: gym.Env):
    q1 = policy['Q1']
    q2 = policy['Q2']
    if np.random.uniform() > 0.5:
        q1, q2 = q2, q1

    rewards = 0
    learning_rate = policy['learning_rate']
    discount = policy['discount']
    done = False

    obs, _ = env.reset()
    action = act(policy, q1, obs)
    q1, q2 = q2, q1

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)

        action_index = np.argmax(q1[next_obs])
        td_error = reward + (discount * q2[next_obs, action_index] - q2[obs, action])
        q2[obs, action] += learning_rate * td_error

        done = terminated or truncated
        obs = next_obs
        action = act(policy, q1, obs)
        q1, q2 = q2, q1

        rewards += reward

    return rewards


def main():
    env = gym.make("FrozenLake-v1")
    log_every = 100
    reward_sum = 9

    policy = create_policy(env.observation_space.n, env.action_space.n, 0.1, 0.1, 0.99)
    for i in range(500000):
        reward_sum += train_episode(policy, env)
        if i % log_every == log_every - 1:
            print(reward_sum / log_every)
            reward_sum = 0

    env = gym.make("FrozenLake-v1", render_mode="human")
    train_episode(policy, env)


if __name__ == '__main__':
    main()
