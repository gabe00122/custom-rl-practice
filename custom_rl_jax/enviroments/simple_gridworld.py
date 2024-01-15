from typing import NamedTuple
from functools import partial
from dataclasses import dataclass, field
import jax
from jax import numpy as jnp, random, Array
from jaxtyping import Float, Int, Bool, Scalar, PRNGKeyArray


class Params(NamedTuple):
    pass


class State(NamedTuple):
    step: Int[Scalar, ""]
    goal: Int[Array, "2"]
    position: Int[Array, "2"]


class Observation(NamedTuple):
    goal: Int[Array, "2"]
    position: Int[Array, "2"]


class Action(NamedTuple):
    direction: Int[Scalar, ""]  # up, right, down, left


class StepReturn(NamedTuple):
    observation: Observation
    state: State
    reward: Float[Scalar, ""]
    done: Bool[Scalar, ""]


class ResetReturn(NamedTuple):
    observation: Observation
    state: State


_directions = jnp.array([
    [0, 1],
    [1, 0],
    [0, -1],
    [-1, 0],
], jnp.int32)


@dataclass(frozen=True)
class SimpleGridWorld:
    width: int = field(default=16)
    height: int = field(default=16)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, rng_key: PRNGKeyArray, state: State, action: Action, params: Params) -> StepReturn:
        direction_vec = _directions[action.direction]
        next_position = state.position + direction_vec
        next_position = jnp.clip(
            next_position,
            jnp.zeros((2,), jnp.int32),
            jnp.array([self.width, self.height], jnp.int32),
        )

        next_state = State(state.step + 1, state.goal, next_position)
        done = jnp.array_equal(next_position, state.goal) | (state.step >= 100)

        observation, state = jax.lax.cond(
            done,
            lambda: self.reset(rng_key, params),
            lambda: ResetReturn(Observation(goal=next_state.goal, position=next_position), next_state),
        )

        return StepReturn(observation, state, jnp.float32(-1.0), done)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng_key: PRNGKeyArray, params: Params) -> ResetReturn:
        rng_goal, rng_pos = random.split(rng_key)

        size = self.width * self.height
        goal_num = random.randint(rng_goal, (), 0, size)
        pos_num = random.randint(rng_pos, (), 0, size - 1)
        pos_num += jax.lax.cond(
            pos_num >= goal_num,
            lambda: 1,
            lambda: 0,
        )

        goal = jnp.array([goal_num % self.width, goal_num // self.width], jnp.int32)
        pos = jnp.array([pos_num % self.width, pos_num // self.width], jnp.int32)

        state = State(step=jnp.int32(0), goal=goal, position=pos)
        observation = Observation(goal, pos)
        return ResetReturn(observation, state)


def main():
    env = SimpleGridWorld()
    rng_key = random.PRNGKey(1)
    params = Params()
    observation, state = env.reset(rng_key, params)
    for _ in range(20):
        action = Action(jnp.int32(0))
        rng_key, step_key = random.split(rng_key)
        observation, state, reward, done = env.step(step_key, state, action, params)
        print(state)


if __name__ == "__main__":
    main()
