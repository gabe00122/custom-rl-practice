from dataclasses import dataclass, field
from functools import partial
from typing import TypedDict
import jax
from jax import numpy as jnp, Array

MetricsRecorderState = TypedDict("MetricsRecorderState", {
    'step': Array,
    'rewards': Array,
    'finished_episode_rewards': Array,
    'ongoing_episode_rewards': Array,
})

MetricsRecorderOutput = TypedDict("MetricsRecorderOutput", {
    'rewards': Array,
})


@dataclass(frozen=True)
class MetricsRecorder:
    total_steps: int = field()
    vec_num: int = field()

    @partial(jax.jit, static_argnums=0)
    def init(self) -> MetricsRecorderState:
        return {
            'step': jnp.array(0, dtype=jnp.int32),
            'rewards': jnp.zeros((self.total_steps,), dtype=jnp.float32),
            'finished_episode_rewards': jnp.zeros((self.vec_num,), dtype=jnp.float32),
            'ongoing_episode_rewards': jnp.zeros((self.vec_num,), dtype=jnp.float32),
        }

    @partial(jax.jit, static_argnums=0)
    def update(self, state: MetricsRecorderState, done: Array, step_rewards: Array) -> MetricsRecorderState:
        step = state['step'] + 1
        ongoing_episode_rewards = state['ongoing_episode_rewards'] + step_rewards

        finished_episode_rewards = jnp.where(done, ongoing_episode_rewards, state['finished_episode_rewards'])
        ongoing_episode_rewards = jnp.where(done, jnp.zeros((self.vec_num,)), ongoing_episode_rewards)

        mean_finished_episode_rewards = jnp.mean(finished_episode_rewards)
        rewards = state['rewards'].at[step].set(mean_finished_episode_rewards)

        return {
            'step': step,
            'rewards': rewards,
            'finished_episode_rewards': finished_episode_rewards,
            'ongoing_episode_rewards': ongoing_episode_rewards,
        }

    @partial(jax.jit, static_argnums=0)
    def get(self, state: MetricsRecorderState) -> MetricsRecorderOutput:
        return {
            'rewards': state['rewards'],
        }

    @partial(jax.jit, static_argnums=0)
    def reset(self, state: MetricsRecorderState) -> MetricsRecorderState:
        return state | {
            'step': jnp.array(0, dtype=jnp.int32),
            'rewards': jnp.zeros((self.total_steps,), dtype=jnp.float32),
        }
