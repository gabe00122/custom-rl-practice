import numpy as np
import pandas as pd
from .metrics_recorder import MetricsRecorderState


class MetricsLoggerNP:
    total_steps: int
    curser: int
    mean_rewards: np.ndarray
    state_value: np.ndarray
    td_error: np.ndarray
    actor_loss: np.ndarray
    critic_loss: np.ndarray
    entropy: np.ndarray

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.curser = 0
        self.mean_rewards = np.zeros(total_steps, dtype=np.float32)
        self.state_value = np.zeros(total_steps, dtype=np.float32)
        self.td_error = np.zeros(total_steps, dtype=np.float32)
        self.actor_loss = np.zeros(total_steps, dtype=np.float32)
        self.critic_loss = np.zeros(total_steps, dtype=np.float32)
        self.entropy = np.zeros(total_steps, dtype=np.float32)

    def log(self, metrics_frame: MetricsRecorderState):
        frame_length = len(metrics_frame['mean_rewards'])
        start = self.curser
        end = self.curser + frame_length

        self.mean_rewards[start:end] = metrics_frame['mean_rewards']
        self.state_value[start:end] = metrics_frame['state_value']
        self.td_error[start:end] = metrics_frame['td_error']
        self.actor_loss[start:end] = metrics_frame['actor_loss']
        self.critic_loss[start:end] = metrics_frame['critic_loss']
        self.entropy[start:end] = metrics_frame['entropy']

        self.curser += frame_length

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'mean_rewards': self.mean_rewards,
            'state_value': self.state_value,
            'td_error': self.td_error,
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,
            'entropy': self.entropy,
        })
