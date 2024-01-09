from dataclasses import dataclass, field
from functools import partial
from typing import TypedDict
import jax
from jax import numpy as jnp, Array
from .finished_reward_recorder import FinishedRewardRecorder, FinishedRewardRecorderState

MetricsRecorderState = TypedDict(
    "MetricsRecorderState",
    {
        "step": Array,
        "mean_rewards": Array,
        "finished_reward_recorder_state": FinishedRewardRecorderState,
    },
)


@dataclass(frozen=True)
class MetricsRecorder:
    step_num: int = field()
    finished_reward_recorder: FinishedRewardRecorder = field()

    @classmethod
    def create(cls, step_num: int, vec_num: int) -> "MetricsRecorder":
        return cls(
            step_num=step_num,
            finished_reward_recorder=FinishedRewardRecorder(vec_num),
        )

    @partial(jax.jit, static_argnums=0)
    def init(self) -> MetricsRecorderState:
        return {
            "step": jnp.int32(0),
            "finished_reward_recorder_state": self.finished_reward_recorder.init(),
            "mean_rewards": jnp.zeros((self.step_num,), dtype=jnp.float32),
        }

    @partial(jax.jit, static_argnums=0)
    def update(self, state: MetricsRecorderState, done: Array, step_rewards: Array) -> MetricsRecorderState:
        step = state["step"]
        finished_reward_recorder_state = state["finished_reward_recorder_state"]
        mean_rewards = state["mean_rewards"]

        finished_reward_recorder_state, finished_rewards = self.finished_reward_recorder.update(
            finished_reward_recorder_state, done, step_rewards
        )

        mean_rewards = mean_rewards.at[step].set(finished_rewards.mean())
        step = step + 1

        return {
            "step": step,
            "finished_reward_recorder_state": finished_reward_recorder_state,
            "mean_rewards": mean_rewards,
        }

    @partial(jax.jit, static_argnums=0)
    def reset(self, state: MetricsRecorderState) -> MetricsRecorderState:
        return state | {
            "step": jnp.int32(0),
            "mean_rewards": jnp.zeros((self.step_num,), dtype=jnp.float32),
        }
