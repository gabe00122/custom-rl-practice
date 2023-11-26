import math
import gymnasium as gym
import numpy as np
from policy import Policy, HyperParams


class NStepSarsa(Policy):
    actions: np.ndarray
    observations: np.ndarray
    rewards: np.ndarray

    def __init__(self, params: HyperParams):
        super().__init__(params)

        self.actions = np.zeros((self.steps + 1,), dtype=np.int32)
        self.observations = np.zeros((self.steps + 1,), dtype=np.int32)
        self.rewards = np.zeros((self.steps + 1,), dtype=np.float64)

    def train_episode(self, env: gym.Env):
        rewards = 0
        weight_update = 1 / (1 - (self.exploration / 2))

        obs, info = env.reset()

        action = self.act(obs)
        self.observations[0] = obs
        self.actions[0] = action

        end_timestep = math.inf
        eval_step = 0
        while True:
            mod_i = eval_step % (self.steps + 1)
            mod_i_inc = (eval_step + 1) % (self.steps + 1)

            if eval_step < end_timestep:
                obs, reward, terminated, truncated, info = env.step(self.actions[mod_i])
                self.rewards[mod_i_inc] = reward
                self.observations[mod_i_inc] = obs
                rewards += reward
                if terminated or truncated:
                    end_timestep = eval_step + 1
                else:
                    self.actions[mod_i_inc] = self.act(obs)
            learning_time_step = eval_step - self.steps + 1
            if learning_time_step >= 0:
                sampling_ratio = 1
                for j in range(learning_time_step + 1, min(learning_time_step + self.steps, end_timestep - 1) + 1):
                    mod_j = j % (self.steps + 1)
                    obs = self.observations[mod_j].item()
                    action = self.actions[mod_j].item()
                    greedy_action = self.greedy_act(obs)
                    if action == greedy_action:
                        sampling_ratio *= weight_update
                    else:
                        sampling_ratio = 0

                estimated_reward = 0
                for j in range(learning_time_step + 1, min(learning_time_step + self.steps, end_timestep) + 1):
                    mod_j = j % (self.steps + 1)
                    estimated_reward += (self.discount ** (j - learning_time_step - 1)) * self.rewards[mod_j].item()

                if learning_time_step + self.steps < end_timestep:
                    mod_rn = (learning_time_step + self.steps) % (self.steps + 1)
                    obs = self.observations[mod_rn].item()
                    action = self.actions[mod_rn].item()
                    estimated_reward += (self.discount ** self.steps) * self.q[obs, action].item()

                mod_timestep = learning_time_step % (self.steps + 1)
                obs = self.observations[mod_timestep].item()
                action = self.actions[mod_timestep].item()
                self.q[obs, action] += self.learning_rate * sampling_ratio * (
                            estimated_reward - self.q[obs, action].item())

            eval_step += 1
            if learning_time_step >= end_timestep - 1:
                break

        return rewards
