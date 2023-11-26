import math
import gymnasium as gym
import numpy as np
from policy import Policy, HyperParams


class TreeBackup(Policy):
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
        steps_inc = self.steps + 1

        obs, info = env.reset()

        action = self.act(obs)
        self.observations[0] = obs
        self.actions[0] = action

        end_timestep = math.inf
        eval_step = 0
        while True:
            mod_step = eval_step % steps_inc
            mod_step_inc = (eval_step + 1) % steps_inc

            if eval_step < end_timestep:
                obs, reward, terminated, truncated, info = env.step(self.actions[mod_step])
                self.rewards[mod_step_inc] = reward
                self.observations[mod_step_inc] = obs
                rewards += reward
                if terminated or truncated:
                    end_timestep = eval_step + 1
                else:
                    self.actions[mod_step_inc] = self.act(obs)

            learning_time_step = eval_step - self.steps + 1
            if learning_time_step >= 0:
                if learning_time_step + 1 >= end_timestep:
                    estimated_reward = self.rewards[end_timestep % steps_inc]
                else:
                    estimated_reward = self.rewards[mod_step_inc] + self.expected_return(
                        self.observations[mod_step_inc].item())

                for k in reversed(range(learning_time_step + 1, min(eval_step, end_timestep - 1) + 1)):
                    mod_k = k % steps_inc
                    obs_k = self.observations[mod_k].item()
                    action_k = self.actions[mod_k].item()
                    reward_k = self.rewards[mod_k]
                    estimated_reward = reward_k + self.discount * (
                            self.expected_return_without(obs_k, action_k)
                            + self.action_probability(obs_k, action_k) * estimated_reward
                    )

                mod_timestep = learning_time_step % steps_inc
                obs = self.observations[mod_timestep]
                action = self.actions[mod_timestep]
                self.q[obs, action] += self.learning_rate * (estimated_reward - self.q[obs, action])

            eval_step += 1
            if learning_time_step >= end_timestep - 1:
                break

        return rewards

    def action_probability(self, obs: int, action: int) -> float:
        if np.argmax(self.q[obs]) == action:
            return 1
        else:
            return 0

    def action_probabilities(self, obs: int) -> np.ndarray:
        greedy = self.greedy_act(obs)
        probs = np.zeros((self.action_space,), dtype=np.float32)
        probs[greedy] = 1
        return probs

    def expected_return(self, obs: int) -> float:
        # probs = self.action_probabilities(obs)
        # return np.sum(self.q[obs] * probs).item()
        return np.max(self.q[obs]).item()

    def expected_return_without(self, obs: int, without_action) -> float:
        # probs = self.action_probabilities(obs)
        # rewards = self.q[obs] * probs
        # rewards[without_action] = 0
        # return np.sum(rewards).item()
        greedy = np.argmax(self.q[obs])
        if without_action == greedy:
            return 0
        else:
            return self.q[obs, greedy].item()


__all__ = ['TreeBackup']
