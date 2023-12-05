import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from policy import Policy
from nstep_sarsa import NStepSarsa
from tree_backup import TreeBackup


def main():
    env = gym.make('FrozenLake-v1', render_mode="human")

    data = {}
    # for step in range(1, 5):
    step = 5
    policy = TreeBackup({
        "action_space": env.action_space.n,
        "observation_space": env.observation_space.n,
        "discount": 0.99,
        "exploration": 0.05,
        "learning_rate": 0.1,
        "steps": step
    })
    data['steps-{}'.format(step)] = train(env, policy, 10000)

    data = pd.DataFrame(data)
    data = data.rolling(100).mean()
    data.plot()
    plt.show()


def train(env: gym.Env, policy: Policy, episodes: int) -> np.ndarray:
    rewards = np.zeros((episodes,))

    for i in tqdm(range(episodes), ncols=140, unit="episode"):
        rewards[i] = policy.train_episode(env)
    return rewards


if __name__ == '__main__':
    main()
