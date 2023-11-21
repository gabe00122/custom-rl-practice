import gymnasium as gym
import numpy as np
from typing_extensions import TypedDict
from hyper_params import HyperParams

Policy = TypedDict('Policy', {'Q': np.ndarray})


def create_policy(state_space: int, action_space: int) -> Policy:
    return {
        'Q': np.zeros((state_space, action_space)),
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
    learning_rate = params['learning_rate']
    discount = params['discount']
    done = False

    obs, _ = env.reset()

    while not done:
        action = act(q, params, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        td_error = reward + (discount * np.max(q[next_obs]) - q[obs, action])
        q[obs, action] += learning_rate * td_error

        done = terminated or truncated
        obs = next_obs
        rewards += reward

    return rewards


def main():
    env = gym.make("CliffWalking-v0")

    params: HyperParams = {
        'exploration': 0.1,
        'discount': 0.99,
        'learning_rate': 0.1
    }
    policy = create_policy(env.observation_space.n, env.action_space.n)
    for i in range(10000):
        train_episode(policy, params, env)
        if i % 100 == 99:
            print(i)

    env = gym.make("CliffWalking-v0", render_mode="human")
    train_episode(policy, params, env)


if __name__ == '__main__':
    main()
