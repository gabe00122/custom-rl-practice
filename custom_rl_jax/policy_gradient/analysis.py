import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

runs = Path("./run").absolute()

all_rewards = pd.DataFrame()

base_lr = 0.0001

# for actor_lr in range(5):
#     for critic_lr in range(5):
# actor_learning_rate = (actor_lr + 1) * base_lr
# critic_learning_rate = (critic_lr + 1) * base_lr

data_frame = pd.read_csv(runs / "log.cvs")
# rewards = data_frame["rewards"]
#total_reward = rewards[990:1000].mean()

# all_rewards[actor_lr, critic_lr] = total_reward
#all_rewards[f'a_{(actor_lr + 1)}-c_{(critic_lr + 1)}'] = rewards

# for i in range(10):
#     data_frame = pd.read_csv(runs / f"{i}" / "data.cvs")
#     all_rewards[i] = data_frame['rewards']
    #rewards = data_frame['rewards']

data_frame = data_frame.rolling(window=10).mean()
#sns.heatmap(all_rewards)
data_frame.plot.line()
plt.show()
