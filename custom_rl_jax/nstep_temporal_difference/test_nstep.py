import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from nstep_sarsa import NStepSarsa
from tree_backup import TreeBackup


def main():
    env = gym.make('FrozenLake-v1')
    policy = TreeBackup({
        "action_space": env.action_space.n,
        "observation_space": env.observation_space.n,
        "discount": 0.99,
        "exploration": 0.05,
        "learning_rate": 0.1,
        "steps": 2
    })

    episodes = 100000
    rewards = np.zeros((episodes, ))

    for i in tqdm(range(episodes), ncols=140, unit="episode"):
        rewards[i] = policy.train_episode(env)

    data = pd.DataFrame({'rewards': rewards})
    data = data.rolling(1000).mean()
    data.plot()
    plt.show()


if __name__ == '__main__':
    main()
