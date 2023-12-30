import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

runs = Path("./run-lander-l2-init-entropy").absolute()

#raw_data = []
#data = np.zeros((8,), dtype=np.float32)

# display_data = pd.DataFrame()

# for i, run in enumerate(runs.iterdir()):
#     with open(run / 'settings.json') as file:
#         settings = json.load(file)
#
#     with open(run / 'metrics.csv') as file:
#         metrics = pd.read_csv(file)
#
#     display_data[f'{settings["env_num"]}'] = metrics['rewards'].rolling(1000).mean()


run_1 = pd.read_csv(runs / 'metrics.csv')
# # take just the losses
run_1.rolling(1000).mean()
run_1 = run_1.iloc[::1000]
#
run_1.plot.line()
plt.show()

# display_data.plot.line()
# plt.show()
