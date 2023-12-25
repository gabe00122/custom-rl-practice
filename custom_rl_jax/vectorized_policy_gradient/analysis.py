import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

runs = Path("./run").absolute()


data_frame_1 = pd.read_csv(runs / "01" / "metrics.csv")
data_frame_2 = pd.read_csv(runs / "05" / "metrics.csv")

data_frame = pd.DataFrame({
    '01': data_frame_1['rewards'],
    '02': data_frame_2['rewards'],
})

data_frame = data_frame.rolling(window=1000).mean()
#sns.heatmap(all_rewards)
data_frame.plot.line()
plt.show()
