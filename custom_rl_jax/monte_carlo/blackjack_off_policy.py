import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from typing_extensions import TypedDict

Policy = TypedDict('Policy', {"exploration": float, "Q": np.ndarray, "C": np.ndarray})
# (player, dealer, ace, action)
Decision = tuple[int, int, int, int]


def create_policy(exploration: float) -> Policy:
    return {
        "exploration": exploration,
        "Q": np.zeros((22, 11, 2, 2), dtype=np.float32),
        "C": np.zeros((22, 11, 2, 2), dtype=np.float32)
    }


def deterministic_act(policy: Policy, player: int, dealer: int, ace: int) -> int:
    q = policy['Q']
    return q[player, dealer, ace].argmax().item()


def exploration_act(policy: Policy, player: int, dealer: int, ace: int) -> int:
    q = policy['Q']
    exploration = policy['exploration']
    upset = exploration > random.random()

    if upset:
        return random.randint(0, 1)

    return q[player, dealer, ace].argmax().item()


def reinforce(policy: Policy, decisions: list[Decision], reward: float):
    q_table = policy['Q']
    c_table = policy['C']
    noise = policy['exploration']

    weight = 1
    weight_update = 1 / (1 - (noise * 0.5))

    for player, dealer, ace, action in reversed(decisions):
        q_value = q_table[player, dealer, ace, action].item()

        c_table[player, dealer, ace, action] += weight
        c = c_table[player, dealer, ace, action].item()

        q_table[player, dealer, ace, action] += (weight / c) * (reward - q_value)

        best_action = q_table[player, dealer, ace].argmax().item()
        if action == best_action:
            weight *= weight_update
        else:
            break

def train(env: gym.Env, policy: Policy):
    decisions: list[Decision] = []
    total_reward = 0.0
    obs, _ = env.reset()
    done = False

    while not done:
        player, dealer, ace = obs
        action = exploration_act(policy, player, dealer, ace)
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
        action = deterministic_act(policy, player, dealer, ace)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


e = gym.make('Blackjack-v1', natural=False, sab=False)
p = create_policy(0.05)

print(p["Q"].size)

results = []

# rewards = 0.0
# for i in range(10000000):
#     index = i % 10000
#     rewards += train(e, p)
#     if index == 9999:
#         reward_mean = rewards / 10000
#         results.append(reward_mean)
#         print(i, reward_mean)
#         rewards = 0.0


plt.plot(results)
plt.show()

# save q table
# np.save('q_off_table.npy', p['Q'])
# np.save('c_off_table.npy', p['C'])

# load q table
p['Q'] = np.load('../q_table.npy')
rewards = 0.0
for i in range(1000000):
    rewards += evaluate(e, p)

print(rewards / 1000000)
