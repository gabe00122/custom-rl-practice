import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

runs = Path("./runs").absolute()

all_rewards = pd.DataFrame()

for i in range(10):
    data_frame = pd.read_csv(runs / f"{i}" / "data.cvs")
    all_rewards[i] = data_frame['rewards']
    #rewards = data_frame['rewards']

all_rewards = all_rewards.rolling(window=100).mean()
all_rewards.plot.line()
plt.show()
