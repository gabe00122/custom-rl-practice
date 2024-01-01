import jax
from jax import numpy as jnp, random, Array
import gymnax
from pathlib import Path
from tqdm import tqdm
from functools import partial
from typing import TypedDict

from .run_settings import RunSettings
from .initialize import create_actor_critic
from .actor_critic import TrainingState

StepState = TypedDict("StepState", {
    "params": TrainingState,
    "key": Array,
    "env_state": Array,
    "importance": Array,
    "obs": Array,
})


def train(settings: RunSettings, save_path: Path):
    total_steps = settings['total_steps']
    env_num = settings['env_num']

    key = random.PRNGKey(settings['seed'])

    env, env_params = gymnax.make(settings['env_name'])
    reset_rng = jax.vmap(env.reset, in_axes=(0, None))
    step_rng = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    action_space = env.action_space(env_params).n
    state_space = env.state_space(env_params).num_spaces - 1

    actor_critic = create_actor_critic(settings, action_space, state_space)

    actor_critic_params, key = actor_critic.init(key)

    key, *env_keys = random.split(key, env_num + 1)
    obs, state = reset_rng(env_keys, env_params)

    act_rng = jax.vmap(actor_critic.act, in_axes=(None, 0, 0))

    step_state: StepState = {
        "params": actor_critic_params,
        "key": key,
        "env_state": state,
        "importance": jnp.ones((env_num,)),
        "obs": obs,
    }

    @jax.jit
    def train_step(step_state: StepState) -> tuple[StepState, Array]:
        params = step_state['params']
        key = step_state['key']
        env_state = step_state['env_state']
        importance = step_state['importance']
        obs = step_state['obs']

        key, *act_keys = random.split(key, env_num + 1)
        actions = act_rng(params, obs, act_keys)

        key, *step_keys = random.split(key, env_num + 1)
        next_obs, env_state, rewards, done, _ = step_rng(env_keys, env_state, actions, env_params)

        params, metrics, importance = actor_critic.train_step(params, obs, actions, rewards, next_obs, done, importance)

        return {
            "params": params,
            "key": key,
            "env_state": env_state,
            "importance": importance,
            "obs": next_obs,
        }

    @jax.jit
    def train_20_steps(step_state: StepState) -> tuple[StepState, Array]:
        # using lax reduce
        step_state = jax.lax.fori_loop(0, 1, lambda i, x: train_step(x), step_state)

        return step_state


    with tqdm(range(total_steps), unit='steps') as tsteps:
        for step in tsteps:
            tsteps.set_description(f"Step {step}")
            step_state = train_20_steps(step_state)

            #tsteps.set_postfix(reward=reward)
