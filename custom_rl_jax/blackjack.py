import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from typing_extensions import TypedDict

# (player, dealer, ace, action)
Decisions = list[tuple[int, int, int, int]]
Policy = TypedDict('Policy', {"noise": float, "Q": np.ndarray, "Q_visits": np.ndarray})


def create_policy(noise: float) -> Policy:
    # (player, dealer, ace, action)
    return {
        "noise": noise,
        "Q": np.zeros((22, 11, 2, 2), dtype=np.float32),
        "Q_visits": np.zeros((22, 11, 2, 2), dtype=np.int32)
    }


def act(policy: Policy, player: int, dealer: int, ace: int) -> int:
    q = policy['Q']
    noise = policy['noise']
    upset = noise > random.random()

    if upset or q[player, dealer, ace, 0] == q[player, dealer, ace, 1]:
        return random.randint(0, 1)

    return q[player, dealer, ace].argmax().item()


def reinforce(policy: Policy, decisions: Decisions, reward: float):
    q_table = policy['Q']
    visits = policy['Q_visits']

    for player, dealer, ace, action in decisions:
        count = visits[player, dealer, ace, action].item()
        q_value = q_table[player, dealer, ace, action].item()

        visits[player, dealer, ace, action] += 1
        q_table[player, dealer, ace, action] = moving_mean(q_value, count, reward)


def moving_mean(mean: float, count: int, sample: float) -> float:
    return mean + ((sample - mean) / (count + 1))


def train(env: gym.Env, policy: Policy):
    decisions: Decisions = []
    total_reward = 0.0
    obs, _ = env.reset()
    done = False

    while not done:
        player, dealer, ace = obs
        action = act(policy, player, dealer, ace)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        decisions.append((player, dealer, ace, action))
        done = terminated or truncated

    reinforce(policy, decisions, total_reward)
    return total_reward


def evaluate(env: gym.Env, policy: Policy):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        player, dealer, ace = obs
        action = act(policy, player, dealer, ace)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


e = gym.make('Blackjack-v1', natural=False, sab=False)
p = create_policy(0.05)

print(p["Q"].size)

results = []


def print_visited():
    visited = (p['Q_visits'] > 1).sum()
    print(visited)
    print(visited / p['Q_visits'].size)


rewards = 0.0
for i in range(10000000):
    index = i % 10000
    rewards += train(e, p)
    if index == 9999:
        reward_mean = rewards / 10000
        results.append(reward_mean)
        print(i, reward_mean)
        rewards = 0.0

    if i % 10000 == 0:
        print_visited()


plt.plot(results)
plt.show()

# save q table
np.save('q_table.npy', p['Q'])
np.save('q_visits.npy', p['Q_visits'])
