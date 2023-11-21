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
    exploration = policy['exploration']
    done = False

    obs, _ = env.reset()
    action = act(policy, obs)

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)

        greedy_return = np.max(q[next_obs])
        exploratory_return = np.mean(q[next_obs])
        expected_return = (greedy_return * (1.0 - exploration)) + (exploratory_return * exploration)

        td_error = reward + (discount * expected_return - q[obs, action])
        q[obs, action] += learning_rate * td_error

        done = terminated or truncated
        obs = next_obs
        action = act(policy, obs)
        rewards += reward

    return rewards


def main():
    env = gym.make("CliffWalking-v0")

    policy = create_policy(env.observation_space.n, env.action_space.n, 0.05, 1.0, 0.99)
    for i in range(50000):
        train_episode(policy, env)
        if i % 100 == 99:
            print(i)

    env = gym.make("CliffWalking-v0", render_mode="human")
    train_episode(policy, env)


if __name__ == '__main__':
    main()
