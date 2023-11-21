import numpy as np
import matplotlib.pyplot as plt
from heatmap import heatmap

q_table = np.load('../q_table.npy')
visits = np.load('../q_visits.npy')
policy_ace = np.flip(q_table[11:, 1:, 1].argmax(axis=2), axis=0)

heatmap(policy_ace, reversed(range(11, 22)), range(1, 11))
#heatmap(visits[:, :, 0].sum(axis=2) > 0, range(4, 26), range(1, 12))
plt.show()
