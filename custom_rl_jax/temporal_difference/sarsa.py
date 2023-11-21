import gymnasium as gym
import numpy as np
from typing_extensions import TypedDict

Policy = TypedDict('Policy', {'Q': np.ndarray, 'exploration': float, 'learning_rate': float, 'discount': float,
                              'state_space': int,
                              'action_space': int})


def create_policy(state_space: int, action_space: int, exploration: float, learning_rate: float,
                  discount: float) -> Policy:
    return {
        'Q': np.zeros((state_space, action_space)),
        'exploration': exploration,
        'learning_rate': learning_rate,
        'discount': discount,
        'state_space': state_space,
        'action_space': action_space
    }


def act(policy: Policy, state: int) -> int:
    if np.random.uniform() < policy['exploration']:
        return np.random.randint(policy['action_space'])
    else:
        return np.argmax(policy['Q'][state])


def train_episode(policy: Policy, env: gym.Env):
    rewards = 0
    q = policy['Q']
    learning_rate = policy['learning_rate']
    discount = policy['discount']
    done = False

    obs, _ = env.reset()
    action = act(policy, obs)

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_action = act(policy, obs)

        td_error = reward + (discount * q[next_obs, next_action] - q[obs, action])
        q[obs, action] += learning_rate * td_error

        done = terminated or truncated
        obs = next_obs
        action = next_action
        rewards += reward

    return rewards


def main():
    env = gym.make("CliffWalking-v0")

    policy = create_policy(env.observation_space.n, env.action_space.n, 0.05, 0.5, 0.99)
    for _ in range(10000):
        train_episode(policy, env)

    env = gym.make("CliffWalking-v0", render_mode="human")
    train_episode(policy, env)


if __name__ == '__main__':
    main()
