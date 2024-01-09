import numpy as np
import pandas as pd
from .metrics_recorder import MetricsRecorderState


class MetricsLoggerNP:
    total_steps: int
    curser: int
    mean_rewards: np.ndarray

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.curser = 0
        self.mean_rewards = np.zeros(total_steps)

    def log(self, metrics_frame: MetricsRecorderState):
        frame_length = len(metrics_frame['mean_rewards'])
        start = self.curser
        end = self.curser + frame_length

        self.mean_rewards[start:end] = metrics_frame['mean_rewards']

        self.curser += frame_length

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'mean_rewards': self.mean_rewards,
        })
