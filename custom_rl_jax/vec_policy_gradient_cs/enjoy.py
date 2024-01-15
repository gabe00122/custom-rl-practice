from pathlib import Path
import json
import orbax.checkpoint as ocp
import jax
from jax import numpy as jnp, random
from jaxtyping import PyTree
from .actor_critic import TrainingState
from .run_settings import RunSettings
from .initialize import create_actor_critic
from ..enviroments.simple_gridworld import SimpleGridWorld, Params
from ..enviroments.simple_gridworld_render import Visualizer
from ..enviroments.encoding import encode_observation, decode_action


def main():
    rng_key = random.PRNGKey(1312)

    settings = load_settings(Path("./multi_search/1/settings.json"))
    env = SimpleGridWorld()
    env_params = Params()

    action_space = 4
    observation_space = 4

    actor_critic = create_actor_critic(settings, action_space, observation_space)

    rng_key, rng_params = random.split(rng_key)
    random_params = actor_critic.init(rng_params)
    # abstract_tree = jax.eval_shape(lambda x: x, random_params)

    actor_critic_params = load_params(Path("./search/1/models"), random_params)
    # actor_critic_params = TrainingState(**actor_critic_params)

    while True:
        state_seq, reward_seq = [], []
        rng_key, rng_reset = random.split(rng_key)
        obs, env_state = env.reset(rng_reset, env_params)
        obs = encode_observation(obs)

        while True:
            state_seq.append(env_state)
            rng_key, rng_act, rng_step = jax.random.split(rng_key, 3)
            action = actor_critic.act(actor_critic_params, obs, rng_act)
            action = decode_action(action)

            next_obs, next_env_state, reward, done = env.step(
                rng_step, env_state, action, env_params
            )
            next_obs = encode_observation(next_obs)

            reward_seq.append(reward)
            if done:
                break
            else:
                obs = next_obs
                env_state = next_env_state

        cum_rewards = jnp.cumsum(jnp.array(reward_seq))
        vis = Visualizer()

        for state in state_seq:
            vis.draw(state)

    # vis.animate("/Users/gabrielkeith/Programming/machine_learning/custom_rl_jax/anim.gif", True)


def load_settings(path: Path) -> RunSettings:
    with open(path, 'r') as file:
        return json.load(file)


def load_params(path: Path, training_state: TrainingState) -> PyTree:
    checkpointer = ocp.PyTreeCheckpointer()

    restore_args = ocp.checkpoint_utils.construct_restore_args(training_state)
    return checkpointer.restore(path.absolute(), item=training_state, restore_args=restore_args)


if __name__ == '__main__':
    main()
