import jax
from jax import numpy as jnp, random
import flax.linen as nn
import gymnasium as gym
from ..networks.mlp import Mlp


def actor_critic(actor_model: nn.Module, critic_model: nn.Module):
    @jax.jit
    def act(actor_params, state: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        return random.categorical(key, logits)

    @jax.jit
    def action_prob(actor_params, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        logits = actor_model.apply(actor_params, state)
        #jax.debug.print("{}", logits)
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

    def train_episode(env: gym.Env, actor_params, critic_params, key: random.PRNGKey):
        discount = 0.99
        actor_learning_rate = 0.0001
        critic_learning_rate = 0.001

        total_reward = 0
        episode_length = 0

        obs, info = env.reset()

        done = False

        i = 1.0

        while not done:
            key, action_key = random.split(key)

            action = act(actor_params, obs, action_key)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            td_error = reward - state_value(critic_params, obs)
            if not done:
                td_error += discount * state_value(critic_params, next_obs)

            critic_params = update_critic(critic_params, obs, critic_learning_rate, td_error)
            actor_params = update_actor(actor_params, obs, action, actor_learning_rate, i, td_error)

            i *= discount
            obs = next_obs

            total_reward += reward
            episode_length += 1

        return actor_params, critic_params, key, total_reward, episode_length

    return train_episode


def main():
    env = gym.make('LunarLander-v2', continuous=False)
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    key = random.PRNGKey(53245)

    actor_model = Mlp(features=[64, 64, action_space])
    critic_model = Mlp(features=[64, 64, 1])

    state_vector = jnp.zeros((state_space,), dtype=jnp.float32)
    key, actor_key, critic_key = random.split(key, 3)
    actor_params = actor_model.init(actor_key, state_vector)
    critic_params = critic_model.init(critic_key, state_vector)

    train_episode = actor_critic(actor_model, critic_model)

    print("started!")

    log_every = 20
    rewards = 0
    for i in range(5000):
        actor_params, critic_params, key, total_reward, episode_length = train_episode(env, actor_params, critic_params, key)
        rewards += total_reward

        if i % log_every == log_every - 1:
            print(f"episode: {i + 1}, reward: {rewards / log_every}")
            rewards = 0

    env = gym.make('LunarLander-v2', continuous=False, render_mode="human")
    for i in range(10):
        actor_params, critic_params, key, total_reward, episode_length = train_episode(env, actor_params, critic_params, key)


if __name__ == '__main__':
    main()
