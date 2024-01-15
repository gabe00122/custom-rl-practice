import gymnax
from gymnax.visualize import Visualizer
from pathlib import Path
import json
from orbax.checkpoint import PyTreeCheckpointer
from typing import Any
import jax
from jax import numpy as jnp, random
from ..actor_critic import TrainingState
from ..run_settings import RunSettings
from ..initialize import create_actor_critic

import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.DEBUG)

def main():
    settings = load_settings(Path("./multi_search/1/settings.json"))
    env, env_params = gymnax.make(settings['env_name'])
    action_space = env.action_space(env_params).n
    observation_space = env.observation_space(env_params).shape[0]

    actor_critic = create_actor_critic(settings, action_space, observation_space)
    actor_critic_params = load_params(Path("./multi_search/1/models"))
    actor_critic_params = TrainingState(**actor_critic_params)

    state_seq, reward_seq = [], []
    rng, rng_reset = random.split(random.PRNGKey(1312))
    obs, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = actor_critic.act(actor_critic_params, obs, rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate("/Users/gabrielkeith/Programming/machine_learning/custom_rl_jax/anim.gif", True)



def load_settings(path: Path) -> RunSettings:
    with open(path, 'r') as file:
        return json.load(file)


def load_params(path: Path) -> Any:
    checkpointer = PyTreeCheckpointer()
    return checkpointer.restore(path.absolute())


if __name__ == '__main__':
    main()
