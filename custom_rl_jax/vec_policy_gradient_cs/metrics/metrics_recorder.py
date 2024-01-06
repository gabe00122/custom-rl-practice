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
        "rewards": Array,
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
            "rewards": jnp.zeros((self.step_num,), dtype=jnp.float32),
            "finished_reward_recorder_state": self.finished_reward_recorder.init(),
        }

    @partial(jax.jit, static_argnums=0)
    def update(self, state: MetricsRecorderState, done: Array, step_rewards: Array) -> MetricsRecorderState:
        step = state["step"]
        rewards = state["rewards"]
        finished_reward_recorder_state = state["finished_reward_recorder_state"]

        finished_reward_recorder_state, finished_reward = self.finished_reward_recorder.update(
            finished_reward_recorder_state, done, step_rewards
        )

        rewards = rewards.at[step].set(finished_reward)
        step = step + 1

        return {
            "step": step,
            "rewards": rewards,
            "finished_reward_recorder_state": finished_reward_recorder_state,
        }

    @partial(jax.jit, static_argnums=0)
    def reset(self, state: MetricsRecorderState) -> MetricsRecorderState:
        return state | {
            "step": jnp.int32(0),
            "rewards": jnp.zeros((self.step_num,), dtype=jnp.float32),
        }
