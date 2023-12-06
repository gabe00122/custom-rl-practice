import jax
from jax import random, numpy as jnp
import flax
from flax import linen as nn
from typing import Sequence
import gymnasium as gym

state_space = 4
action_space = 2
exploration = 0.01


def mish(x: jnp.ndarray):
    return x * jnp.tanh(nn.softplus(x))


class Mlp(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            not_last_feat = i != len(self.features) - 1
            initializer_scale = 1.0 if not_last_feat else 0.01

            x = nn.Dense(
                feat,
                name=f'Layer {i}',
                kernel_init=nn.initializers.variance_scaling(initializer_scale, 'fan_avg', 'uniform'),
                bias_init=nn.initializers.constant(0.01)
            )(x)
            if not_last_feat:
                x = mish(x)
        return x


seed = random.key(789796796789)
print(f"seed {seed}")
key, policy_key = random.split(seed, 2)

policy_model = Mlp(features=[64, 64, action_space])
policy_params = policy_model.init(policy_key, jnp.zeros((state_space,), dtype=jnp.float32))

params = {
    'policy_params': policy_params,
    'exploration': exploration,
    'discount': 0.99,
    'learning_rate': 0.1,
}


@jax.jit
def act(policy_params, exploration, state: jnp.ndarray, key: random.PRNGKey):
    cond_key, decision_key = random.split(key)
    return jax.lax.cond(random.uniform(cond_key) < exploration,
                        lambda: random.randint(decision_key, (), 0, action_space),
                        lambda: jnp.argmax(policy_model.apply(policy_params, state))
                        )


@jax.jit
def q(policy_params, state: jnp.ndarray, action):
    return policy_model.apply(policy_params, state)[action]


q_grad = jax.grad(q)


@jax.jit
def learn(policy_params, obs, action, next_obs, next_action, reward, discount, learning_rate):
    td_error = reward + discount * q(policy_params, next_obs, next_action) - q(policy_params, obs, action)
    td_step = learning_rate * td_error

    policy_grad = q_grad(policy_params, obs, action)
    return jax.tree_util.tree_map(
        lambda weight, grad: weight + td_step * grad,
        policy_params,
        policy_grad
    )


@jax.jit
def learn_end(policy_params, obs, action, reward, learning_rate):
    td_error = reward - q(policy_params, obs, action)
    td_step = learning_rate * td_error

    policy_grad = q_grad(policy_params, obs, action)
    return jax.tree_util.tree_map(
        lambda weight, grad: weight + td_step * grad,
        policy_params,
        policy_grad
    )


@jax.jit
def act_learn(policy_params, obs, action, next_obs, reward, discount, learning_rate, exploration, key):
    next_action = act(policy_params, exploration, next_obs, key)
    next_policy = learn(policy_params, obs, action, next_obs, next_action, reward, discount, learning_rate)
    return next_action, next_policy


def train_episode(env: gym.Env, params, key):
    rewards = 0
    policy_params = params['policy_params']
    learning_rate = params['learning_rate']
    discount = params['discount']
    exploration = params['exploration']

    obs, _ = env.reset()
    done = False

    key, action_key = random.split(key)
    action = act(policy_params, exploration, obs, action_key)

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        reward /= 200
        rewards += reward

        done = terminated or truncated

        if done:
            policy_params = learn_end(policy_params, obs, action, reward, learning_rate)
            break

        key, action_key = random.split(key)

        next_action, policy_params = act_learn(policy_params, obs, action, next_obs, reward, discount, learning_rate,
                                               exploration, action_key)

        obs = next_obs
        action = next_action

    return {'policy_params': policy_params, 'discount': discount, 'exploration': exploration,
            'learning_rate': learning_rate}, rewards


print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(policy_params)))
# print(policy_params)

env = gym.make('CartPole-v1')  # , render_mode="human")
# env = gym.wrappers.NormalizeObservation(env)
# env = gym.wrappers.NormalizeReward(env)

cum_rewards = 0
print_every = 100

for i in range(10000):
    key, episode_key = random.split(key)
    params, rewards = train_episode(env, params, episode_key)
    cum_rewards += rewards

    if i % print_every == print_every - 1:
        print(cum_rewards / print_every)
        # print(policy_model.apply(params['policy_params'], jnp.array([0, 0, 0, 0])))
        print(f"episode {i}")
        cum_rewards = 0

env = gym.make('CartPole-v1', render_mode="human")

for _ in range(10):
    key, episode_key = random.split(key)
    params, rewards = train_episode(env, params, episode_key)
