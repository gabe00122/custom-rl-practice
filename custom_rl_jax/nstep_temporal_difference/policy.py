import numpy as np
import gymnasium as gym
from abc import ABCMeta, abstractmethod
from typing_extensions import TypedDict


HyperParams = TypedDict('HyperParams', {
    'observation_space': int,
    'action_space': int,
    'exploration': float,
    'steps': int,
    'discount': float,
    'learning_rate': float
})


class Policy(metaclass=ABCMeta):
    observation_space: int
    action_space: int
    q: np.ndarray
    exploration: float
    steps: int
    discount: float
    learning_rate: float

    def __init__(self, params: HyperParams):
        self.observation_space = params['observation_space']
        self.action_space = params['action_space']
        self.exploration = params['exploration']
        self.steps = params['steps']
        self.discount = params['discount']
        self.learning_rate = params['learning_rate']
        self.q = np.random.rand(self.observation_space, self.action_space) * 0.05

    def act(self, observation: int) -> int:
        if np.random.uniform() > self.exploration:
            return self.greedy_act(observation)
        else:
            return self.exploratory_act(observation)

    def greedy_act(self, observation: int) -> int:
        return np.argmax(self.q[observation])

    def exploratory_act(self, observation: int) -> int:
        return np.random.randint(self.action_space)

    @abstractmethod
    def train_episode(self, env: gym.Env) -> float: pass


__all__ = ['Policy', 'HyperParams']
