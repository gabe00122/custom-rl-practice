import jax
from jax import numpy as jnp, random
import flax.linen as nn
import gymnasium as gym
from ..networks.mlp import Mlp
from typing import Any
from typing_extensions import TypedDict

from pathlib import Path
import shutil

from tqdm import tqdm
import orbax.checkpoint as ocp
import pandas as pd
import numpy as np

HyperParams = TypedDict('HyperParams', {
    'discount': float,
    'actor_learning_rate': float,
    'critic_learning_rate': float,
})

Metrics = TypedDict('Metrics', {
    'reward': float,
    'length': int,
    'state_value': float,
    'td_error': float
})


def actor_critic(actor_model: nn.Module, critic_model: nn.Module):
    @jax.jit
    def act(actor_params, state: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        return random.categorical(key, logits)

    @jax.jit
    def action_prob(actor_params, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        return nn.softmax(logits)[action]

    @jax.jit
    def ln_action_prob(actor_params, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(action_prob(actor_params, state, action))

    grad_ln_action_prob = jax.grad(ln_action_prob)

    @jax.jit
    def state_value(critic_params, state: jnp.ndarray) -> jnp.ndarray:
        return critic_model.apply(critic_params, state).reshape(())

    grad_state_value = jax.grad(state_value)

    @jax.jit
    def update_critic(critic_params, state, critic_learning_rate, td_error):
        step_size = critic_learning_rate * td_error

        critic_gradient = grad_state_value(critic_params, state)
        return jax.tree_map(lambda weight, grad: weight + step_size * grad,
                            critic_params,
                            critic_gradient)

    @jax.jit
    def update_actor(actor_params, state, action, actor_learning_rate, i, td_error):
        step_size = actor_learning_rate * i * td_error

        policy_gradient = grad_ln_action_prob(actor_params, state, action)
        return jax.tree_map(lambda weight, grad: weight + step_size * grad,
                            actor_params,
                            policy_gradient)

    def train_episode(env: gym.Env, actor_params: dict[str, Any], critic_params: dict[str, Any],
                      hyper_params: HyperParams, key: random.PRNGKey):
        discount = hyper_params['discount']
        actor_learning_rate = hyper_params['actor_learning_rate']
        critic_learning_rate = hyper_params['critic_learning_rate']

        metrics: Metrics = {
            'td_error': 0,
            'state_value': 0,
            'reward': 0,
            'length': 0
        }

        obs, info = env.reset()

        done = False

        i = 1.0

        while not done:
            key, action_key = random.split(key)

            action = act(actor_params, obs, action_key)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            s_value = state_value(critic_params, obs)
            td_error = reward - s_value
            if not done:
                td_error += discount * state_value(critic_params, next_obs)

            critic_params = update_critic(critic_params, obs, critic_learning_rate, td_error)
            actor_params = update_actor(actor_params, obs, action, actor_learning_rate, i, td_error)

            i *= discount
            obs = next_obs

            metrics['state_value'] += s_value
            metrics['td_error'] += td_error
            metrics['reward'] += reward
            metrics['length'] += 1

        return actor_params, critic_params, key, metrics

    return train_episode


def main():
    hyper_params: HyperParams = {
        'discount': 0.99,
        'actor_learning_rate': 0.0001,
        'critic_learning_rate': 0.001,
    }

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    key = random.PRNGKey(53245)

    actor_model = Mlp(features=[64, 64, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)

    state_vector = jnp.zeros((state_space,), dtype=jnp.float32)

    for run in range(10):
        key, actor_key, critic_key = random.split(key, 3)
        actor_params = actor_model.init(actor_key, state_vector)
        critic_params = critic_model.init(critic_key, state_vector)

        train_episode = actor_critic(actor_model, critic_model)

        print("started!")
        total_episodes = 3000
        rewards = np.zeros((total_episodes,))
        state_values = np.zeros((total_episodes,))
        td_errors = np.zeros((total_episodes,))

        with tqdm(range(total_episodes), unit='episode', delay=1.0) as tepisode:
            for episode in tepisode:
                tepisode.set_description(f"Episode {episode}")

                actor_params, critic_params, key, metrics = train_episode(env, actor_params, critic_params, hyper_params,
                                                                          key)
                rewards[episode] = metrics['reward']
                state_values[episode] = metrics['state_value']
                td_errors[episode] = metrics['td_error']

                tepisode.set_postfix(reward=metrics['reward'], state_value=round(metrics['state_value'] / metrics['reward']))

        # write hyperparameters
        run_path = Path(f"./runs/{run}").absolute()

        # save model
        checkpoint_path = Path(run_path)
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        params = {
            'actor': actor_params,
            'critic': critic_params,
        }
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(checkpoint_path / "checkpoint", params)

        # save metrics
        data_frame = pd.DataFrame({
            'td-errors': td_errors,
            'state-values': state_values,
            'rewards': rewards,
        })

        data_path = Path(run_path).absolute()
        data_path.mkdir(parents=True, exist_ok=True)
        data_frame.to_csv(data_path / "data.cvs", index=False)


if __name__ == '__main__':
    main()
