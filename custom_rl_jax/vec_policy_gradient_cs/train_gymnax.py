import jax
from jax import numpy as jnp, random, Array
import gymnax
from pathlib import Path
from tqdm import tqdm
from typing import TypedDict

from .run_settings import RunSettings
from .initialize import create_actor_critic
from .actor_critic import TrainingState
from .metrics.metrics_recorder import MetricsRecorder, MetricsRecorderState

StepState = TypedDict("StepState", {
    "params": TrainingState,
    "key": Array,
    "env_state": Array,
    "importance": Array,
    "obs": Array,
    "metrics_recorder_state": MetricsRecorderState,
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

    key, actor_critic_key = random.split(key)
    actor_critic_params = actor_critic.init(actor_critic_key)

    key, env_key = random.split(key)
    env_keys = random.split(env_key, env_num)
    obs, state = reset_rng(env_keys, env_params)

    act_rng = jax.vmap(actor_critic.act, in_axes=(None, 0, 0))

    steps_per_update = 100
    metrics_recorder = MetricsRecorder.create(steps_per_update, env_num)

    step_state: StepState = {
        "params": actor_critic_params,
        "key": key,
        "env_state": state,
        "importance": jnp.ones((env_num,)),
        "obs": obs,
        "metrics_recorder_state": metrics_recorder.init(),
    }

    @jax.jit
    def train_step(step_state: StepState) -> StepState:
        params = step_state['params']
        key = step_state['key']
        env_state = step_state['env_state']
        importance = step_state['importance']
        obs = step_state['obs']
        metrics_recorder_state = step_state['metrics_recorder_state']

        keys = random.split(key, env_num + 1)
        key = keys[0]
        act_keys = keys[1:]

        actions = act_rng(params, obs, act_keys)

        keys = random.split(key, env_num + 1)
        key = keys[0]
        env_keys = keys[1:]
        next_obs, env_state, rewards, done, _ = step_rng(env_keys, env_state, actions, env_params)

        params, metrics, importance = actor_critic.train_step(params, obs, actions, rewards, next_obs, done, importance)

        metrics_recorder_state = metrics_recorder.update(metrics_recorder_state, done, rewards)

        return {
            "params": params,
            "key": key,
            "env_state": env_state,
            "importance": importance,
            "obs": next_obs,
            "metrics_recorder_state": metrics_recorder_state,
        }

    @jax.jit
    def train_n_steps(step_state: StepState) -> StepState:
        # using lax reduce
        return jax.lax.scan(lambda s, _: (train_step(s), None), step_state, None, length=steps_per_update)[0]

    with tqdm(range(total_steps // steps_per_update), unit='steps', unit_scale=steps_per_update) as tsteps:
        for step in tsteps:
            tsteps.set_description(f"Step {step}")
            step_state = train_n_steps(step_state)

            metrics_recorder_state = step_state['metrics_recorder_state']
            mean_reward = jnp.mean(step_state['metrics_recorder_state']['rewards'])
            step_state |= {"metrics_recorder_state": metrics_recorder.reset(metrics_recorder_state)}

            tsteps.set_postfix(reward=mean_reward)
